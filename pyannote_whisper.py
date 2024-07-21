import whisper
import os
from pyannote.audio import Pipeline
from pyannote_whisper.utils import diarize_text
from dotenv import load_dotenv
import yaml
import time 

# Load YAML configuration
with open("config.yaml", 'r') as stream:
    try:
        config = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Get filenames from the configuration
audio_input_file = config['filenames']['audio_input_file']
output_file = config['filenames']['output_file']
whisper_model = config['modelnames']['whisper']
pyannote_model = config['modelnames']['pyannote']
include_timestamps = config['output']['include_timestamps']

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

start_time_translation = time.time()

pipeline = Pipeline.from_pretrained(
  pyannote_model,
  use_auth_token=os.getenv("HUGGINGFACE_TOKEN"))

model = whisper.load_model(whisper_model)
asr_result = model.transcribe(audio_input_file)

# Calculate and print the elapsed time for translating text
elapsed_time_translation = time.time() - start_time_translation

# Start measuring time for separating each speaker
start_time_speaker_separation = time.time()

diarization_result = pipeline(audio_input_file)
final_result = diarize_text(asr_result, diarization_result)

for seg, spk, sent in final_result:
    print(f'{seg.start:.2f} {seg.end:.2f} {spk} {sent}')

# Determine unique speakers
unique_speakers = set([spk for _, spk, _ in final_result])

# Calculate and print the elapsed time for separating speakers
elapsed_time_speaker_separation = time.time() - start_time_speaker_separation
print(f"Time taken to translate the text: {elapsed_time_translation:.2f} seconds")
print(f"Time taken to separate speakers: {elapsed_time_speaker_separation:.2f} seconds")
print(f"Total time: {(elapsed_time_speaker_separation + elapsed_time_translation):.2f} seconds")

# Ask user for speaker names
speaker_names = []
for speaker in unique_speakers:
    speaker_name = input(f"Enter name for speaker {speaker}: ")
    speaker_names.append(speaker_name)

# Write results to output file with speaker names
with open(output_file, 'w') as f:
    for seg, spk, sent in final_result:
        speaker_index = list(unique_speakers).index(spk)
        speaker_name = speaker_names[speaker_index]
        if include_timestamps:
            line = f'{seg.start:.2f} {seg.end:.2f} {speaker_name} {sent}\n'
        else:
            line = f'{speaker_name} {sent}\n'
        f.write(line)