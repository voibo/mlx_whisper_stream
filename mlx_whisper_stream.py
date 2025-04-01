#!/usr/bin/env python3
import sys
import json
import numpy as np
import signal
import torch
import mlx_whisper
import mlx.core as mx
import logging
import os
import datetime
from silero_vad import load_silero_vad
from silero_vad_iterator import FixedVADIterator

DEBUG = False

# Logger configuration
logger = logging.getLogger("mlx_whisper_stream")
logger.setLevel(logging.DEBUG if DEBUG else logging.INFO)

# Handler for standard error output
error_handler = logging.StreamHandler(sys.stderr)
error_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(error_handler)

# Constants and configuration
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DEBUG_AUDIO_PATH = f"{BASE_PATH}/output/"
SAMPLE_RATE = 16000

# Counter and output directory for debugging
debug_counter = 0
debug_output_dir = None

should_exit = False
def signal_handler(signum, frame):
    global should_exit
    should_exit = True

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# Load Silero VAD model and initialize FixedVADIterator
model = load_silero_vad()
vad_iterator = FixedVADIterator(
    model,
    threshold=0.5,
    sampling_rate=SAMPLE_RATE,
    min_silence_duration_ms=500,
    speech_pad_ms=100
)

'''
result structure:
{
  "text": "いろいろあったと思うんですけれども",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 2.7800000000000002,
      "text": "いろいろあったと思うんですけれども",
      "tokens": [
        50365, 45, 45196, 20, 5157, 11016, 1764, 12488, 1764, 12488, 3590,
        10102, 22372, 17760, 4767, 39256, 41397, 50504
      ],
      "temperature": 0.0,
      "avg_logprob": -0.12594705698441486,
      "compression_ratio": 1.6996587030716723,
      "no_speech_prob": 9.634418263182454e-12
    },
    // ... other segments ...
  ],
  "language": "ja"
}
'''

def process_speech_buffer(buffer, speech_start_time, speech_end_time):
    global debug_counter, debug_output_dir
    
    if buffer:
        full_waveform = torch.cat(buffer, dim=0).unsqueeze(0)
        
        # File output in debug mode
        if DEBUG and debug_output_dir:
            try:
                # Save buffer's raw data as is
                audio_data = full_waveform.numpy()
                filename = f"{debug_output_dir}/segment_{debug_counter:04d}_{speech_start_time:.2f}_{speech_end_time:.2f}.npy"
                np.save(filename, audio_data)
                logger.debug(f"Saved raw audio buffer to {filename}")
                debug_counter += 1
            except Exception as e:
                logger.error(f"Failed to save debug audio: {str(e)}")
        
        # Debug log
        logger.debug(f"Processing buffer with duration: {speech_end_time - speech_start_time:.2f}s")

        audio_array = mx.array(full_waveform.numpy()).flatten().astype(mx.float32)
        try:
            result = mlx_whisper.transcribe(
                audio_array,
                path_or_hf_repo=f"{BASE_PATH}/model/whisper-large-v3-turbo",
                language="ja",
                initial_prompt="宜しくお願いします。",
            )
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            return []
            
        if result and "segments" in result:
            # Calculate the speech duration in seconds
            speech_duration = speech_end_time - speech_start_time
            whisper_duration = sum(seg["end"] - seg["start"] for seg in result["segments"])
            
            # Calculate scaling factor if whisper duration differs from actual duration
            scale_factor = speech_duration / whisper_duration if whisper_duration > 0 else 1.0
            
            for i, seg in enumerate(result["segments"]):
                # Adjust segment timestamps
                original_start = seg["start"]
                original_end = seg["end"]
                
                # Scale and offset timestamps
                seg["start"] = speech_start_time + (original_start * scale_factor)
                seg["end"] = speech_start_time + (original_end * scale_factor)
                
                # Ensure end doesn't exceed actual speech end time
                if seg["end"] > speech_end_time:
                    seg["end"] = speech_end_time
                
                if (seg["start"] is not None and 
                    seg["end"] is not None and 
                    seg["text"] is not None and 
                    seg["text"] != ""):
                    data = {
                        "start": seg["start"],
                        "end": seg["end"],
                        "text": seg["text"],
                    }
                    # Always output results to stdout
                    sys.stdout.write(json.dumps(data) + "\n")
                    logger.debug(f"Generated segment: {data}")
                else:
                    logger.warning(f"Missing data in segment: {seg}")
            sys.stdout.flush()
        else:
            logger.warning("No segments found in transcription result")

        return []  # Reset buffer
    return buffer

def main():
    global debug_output_dir
    
    # Create output directory in DEBUG mode
    if DEBUG:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_output_dir = os.path.join(DEBUG_AUDIO_PATH, f"session_{timestamp}")
        try:
            os.makedirs(debug_output_dir, exist_ok=True)
            logger.debug(f"Created debug output directory: {debug_output_dir}")
        except Exception as e:
            logger.error(f"Failed to create debug directory: {str(e)}")
            debug_output_dir = None
    
    chunk_size = SAMPLE_RATE * 4  # float32: 4 bytes/sample

    # Variables for speech detection
    speech_buffer = []  # List to accumulate float32 tensors during speech
    is_speech_active = False  # Flag to track if speech is currently active
    speech_start_time = 0  # Timestamp when speech starts (in seconds)
    speech_start_sample = 0  # Sample index when speech starts
    current_sample = 0  # Current sample index
    
    # Variables for recording raw_audio every 10 seconds
    last_raw_dump_time = 0
    raw_dump_interval = 10  # 10 seconds interval
    
    logger.info("Starting audio processing")
    
    while not should_exit:
        # Read audio chunk from stdin
        raw_audio: bytes = sys.stdin.buffer.read(chunk_size)
        if not raw_audio:
            break
        
        audio_float32 = torch.from_numpy(
            np.frombuffer(raw_audio, dtype=np.float32).copy()  # Copy for writing safety
        )
        
        # Update current sample position
        current_sample += len(audio_float32)
        current_time = current_sample / SAMPLE_RATE

        # Record raw_audio every 10 seconds in DEBUG mode
        if DEBUG and debug_output_dir and (current_time - last_raw_dump_time >= raw_dump_interval):
            try:
                # Save binary data as is
                raw_filename = f"{debug_output_dir}/raw_{int(current_time):04d}.bin"
                with open(raw_filename, "wb") as f:
                    f.write(raw_audio)
                logger.debug(f"Saved raw input data to {raw_filename}")
                last_raw_dump_time = current_time  # Update last saved time
            except Exception as e:
                logger.error(f"Failed to save raw input data: {str(e)}")

        # Detect speech segments
        vad_result = vad_iterator(audio_float32.numpy(), return_seconds=False)
        
        if vad_result is not None:
            if 'start' in vad_result:
                # Speech start detected
                is_speech_active = True
                # Calculate actual start time in seconds
                speech_start_sample = vad_result['start']
                speech_start_time = speech_start_sample / SAMPLE_RATE
                # Add audio to buffer
                speech_buffer.append(audio_float32)
                logger.debug(f"Speech start detected at {speech_start_time:.2f}s")
                
            if 'end' in vad_result:
                # Speech end detected
                is_speech_active = False
                # Calculate actual end time in seconds
                speech_end_sample = vad_result['end']
                speech_end_time = speech_end_sample / SAMPLE_RATE
                # Include current chunk in processing
                speech_buffer.append(audio_float32)
                logger.debug(f"Speech end detected at {speech_end_time:.2f}s")
                # Process the buffer as speech has ended
                speech_buffer = process_speech_buffer(speech_buffer, speech_start_time, speech_end_time)
        elif is_speech_active:
            # Continue adding to buffer if speech is active
            speech_buffer.append(audio_float32)
    
    # Process any remaining buffer content
    if is_speech_active and speech_buffer:
        speech_end_time = current_time  # Use current time as end time for last segment
        process_speech_buffer(speech_buffer, speech_start_time, speech_end_time)

    # Clean up resources
    vad_iterator.reset_states()
    logger.info("Audio processing completed")
    sys.exit(0)

if __name__ == '__main__':
    main()
