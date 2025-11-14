# app.py
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import uuid
import threading
from datetime import datetime
import subprocess
import logging
from pyngrok import ngrok
from wan.utils.multitalk_utils import save_video_ffmpeg

# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import argparse
import warnings

warnings.filterwarnings('ignore')

import random

import torch
import torch.distributed as dist
from PIL import Image
# import subprocess

# import wan
# from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
# from wan.utils.utils import cache_image, cache_video, str2bool
# from wan.utils.multitalk_utils import save_video_ffmpeg
# from kokoro import KPipeline
from transformers import Wav2Vec2FeatureExtractor
from src.audio_analysis.wav2vec2 import Wav2Vec2Model

import librosa
import pyloudnorm as pyln
import numpy as np
from einops import rearrange
import soundfile as sf
import re
from kokoro import KPipeline
from src.audio_analysis.wav2vec2 import Wav2Vec2Model

wan = None
WAN_CONFIGS = None

# Flask app setup
app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Global variables for model
wan_pipeline = None
wav2vec_feature_extractor = None
audio_encoder = None
device = "cuda:0" if torch.cuda.is_available() else "cpu"

_models_cache = {
    'wan_pipeline': None,
    'wav2vec_feature_extractor': None,
    'audio_encoder': None,
    'initialized': False
}


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def custom_init(device, wav2vec_dir):    
    audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec_dir, local_files_only=True).to(device)
    audio_encoder.feature_extractor._freeze_parameters()
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_dir, local_files_only=True)
    return wav2vec_feature_extractor, audio_encoder

def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio

def get_embedding(speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000, device='cpu'):
    """Extract audio embeddings with JSON logging"""
    try:
        audio_duration = len(speech_array) / sr
        video_length = audio_duration * 25
        
        logger.info(f"\n🔍 EMBEDDING EXTRACTION")
        logger.info(f"   Audio duration: {audio_duration:.2f}s")
        logger.info(f"   Video length: {video_length:.0f} frames")
        logger.info(f"   Speech array shape: {speech_array.shape}")
        
        audio_feature = np.squeeze(
            wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
        )
        audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
        audio_feature = audio_feature.unsqueeze(0)
        
        logger.info(f"   Audio feature shape: {audio_feature.shape}")
        
        with torch.no_grad():
            embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)
        
        if len(embeddings) == 0:
            logger.error("❌ Failed to extract embedding")
            return None
        
        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")
        audio_emb = audio_emb.cpu().detach()
        
        embedding_info = {
            "shape": str(audio_emb.shape),
            "dtype": str(audio_emb.dtype),
            "dimensions": {
                "sequence_length": audio_emb.shape[0],
                "batch_size": audio_emb.shape[1],
                "embedding_dim": audio_emb.shape[2]
            }
        }
        log_json("✅ EMBEDDING EXTRACTED", embedding_info)
        
        return audio_emb
        
    except Exception as e:
        logger.error(f"❌ Error in get_embedding: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def extract_audio_from_video(filename, sample_rate):
    raw_audio_path = filename.split('/')[-1].split('.')[0] + '.wav'
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i",
        str(filename),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "2",
        str(raw_audio_path),
    ]
    subprocess.run(ffmpeg_command, check=True)
    human_speech_array, sr = librosa.load(raw_audio_path, sr=sample_rate)
    human_speech_array = loudness_norm(human_speech_array, sr)
    os.remove(raw_audio_path)

    return human_speech_array

def audio_prepare_single(audio_path, sample_rate=16000):
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in ['.mp4', '.mov', '.avi', '.mkv']:
        human_speech_array = extract_audio_from_video(audio_path, sample_rate)
        return human_speech_array
    else:
        human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)
        human_speech_array = loudness_norm(human_speech_array, sr)
        return human_speech_array

def process_tts_single(text, save_dir, voice1):    
    s1_sentences = []
    pipeline = KPipeline(lang_code='a', repo_id='/content/drive/MyDrive/weights/Kokoro-82M')
    voice_tensor = torch.load(voice1, weights_only=True)
    generator = pipeline(
        text, voice=voice_tensor,
        speed=1, split_pattern=r'\n+'
    )
    audios = []
    for i, (gs, ps, audio) in enumerate(generator):
        audios.append(audio)
    if not audios:
        print("❌ No audio generated")
        return np.zeros(16000), os.path.join(save_dir, 's1.wav')
    audios = torch.concat(audios, dim=0)
    s1_sentences.append(audios)
    s1_sentences = torch.concat(s1_sentences, dim=0)
    save_path1 = f'{save_dir}/s1.wav'
    sf.write(save_path1, s1_sentences, 24000)
    s1, _ = librosa.load(save_path1, sr=16000)
    return s1, save_path1

def process_tts_multi(text, save_dir, voice1, voice2):
    print(f"🎤 Processing 2-speaker TTS: {text}")
    pattern = r'\(s(\d+)\)\s*([^()]*?)(?=\(s\d+\)|$)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    if not matches:
        print("❌ No speaker markers found. Treating as single speaker...")
        return process_tts_single(text, save_dir, voice1)
    
    s1_sentences = []
    s2_sentences = []
    pipeline = KPipeline(lang_code='a', repo_id='/content/drive/MyDrive/weights/Kokoro-82M')
    
    for idx, (speaker, content) in enumerate(matches):
        content = content.strip()
        if not content:
            continue
            
        print(f"  Speaker {speaker}: '{content}'")
        
        try:
            voice_tensor = torch.load(voice1 if speaker == '1' else voice2, weights_only=True)
            generator = pipeline(content, voice=voice_tensor, speed=1, split_pattern=r'\n+')
            audios = [audio for i, (gs, ps, audio) in enumerate(generator)]
            
            if audios:
                combined = torch.concat(audios, dim=0)
                if speaker == '1':
                    s1_sentences.append(combined)
                    s2_sentences.append(torch.zeros_like(combined))
                else:
                    s2_sentences.append(combined)
                    s1_sentences.append(torch.zeros_like(combined))
        except Exception as e:
            print(f"⚠️ Error for speaker {speaker}: {e}")
    
    if not s1_sentences or not s2_sentences:
        print("❌ No audio generated for speakers")
        s1 = np.zeros(16000)
        s2 = np.zeros(16000)
        sum_path = os.path.join(save_dir, 'sum.wav')
        sf.write(sum_path, s1, 16000)
        return s1, s2, sum_path
    
    s1_combined = torch.concat(s1_sentences, dim=0) if s1_sentences else torch.tensor([])
    s2_combined = torch.concat(s2_sentences, dim=0) if s2_sentences else torch.tensor([])
    
    max_len = max(len(s1_combined), len(s2_combined), 1)
    
    if len(s1_combined) < max_len:
        s1_combined = torch.cat([s1_combined, torch.zeros(max_len - len(s1_combined))])
    if len(s2_combined) < max_len:
        s2_combined = torch.cat([s2_combined, torch.zeros(max_len - len(s2_combined))])
    
    sum_combined = s1_combined + s2_combined
    
    save_path1 = f'{save_dir}/s1.wav'
    save_path2 = f'{save_dir}/s2.wav'
    save_path_sum = f'{save_dir}/sum.wav'
    
    sf.write(save_path1, s1_combined.numpy(), 24000)
    sf.write(save_path2, s2_combined.numpy(), 24000)
    sf.write(save_path_sum, sum_combined.numpy(), 24000)
    
    s1, _ = librosa.load(save_path1, sr=16000)
    s2, _ = librosa.load(save_path2, sr=16000)
    
    print(f"✅ 2-speaker audio generated")
    return s1, s2, save_path_sum

def process_tts_triple(text, save_dir, voice1, voice2, voice3):
    print(f"🎤 Processing 3-speaker TTS: {text}")
    pattern = r'\(s(\d+)\)\s*([^()]*?)(?=\(s\d+\)|$)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    if not matches:
        print("❌ No speaker markers found. Treating as single speaker...")
        audio, path = process_tts_single(text, save_dir, voice1)
        return audio, np.zeros(16000), np.zeros(16000), path
    
    s1_sentences = []
    s2_sentences = []
    s3_sentences = []
    pipeline = KPipeline(lang_code='a', repo_id='/content/drive/MyDrive/weights/Kokoro-82M')
    
    for idx, (speaker, content) in enumerate(matches):
        content = content.strip()
        if not content:
            continue
            
        print(f"  Speaker {speaker}: '{content}'")
        
        try:
            if speaker == '1':
                voice_tensor = torch.load(voice1, weights_only=True)
            elif speaker == '2':
                voice_tensor = torch.load(voice2, weights_only=True)
            elif speaker == '3':
                voice_tensor = torch.load(voice3, weights_only=True)
            else:
                continue
            
            generator = pipeline(content, voice=voice_tensor, speed=1, split_pattern=r'\n+')
            audios = [audio for i, (gs, ps, audio) in enumerate(generator)]
            
            if audios:
                combined = torch.concat(audios, dim=0)
                if speaker == '1':
                    s1_sentences.append(combined)
                    s2_sentences.append(torch.zeros_like(combined))
                    s3_sentences.append(torch.zeros_like(combined))
                elif speaker == '2':
                    s2_sentences.append(combined)
                    s1_sentences.append(torch.zeros_like(combined))
                    s3_sentences.append(torch.zeros_like(combined))
                elif speaker == '3':
                    s3_sentences.append(combined)
                    s1_sentences.append(torch.zeros_like(combined))
                    s2_sentences.append(torch.zeros_like(combined))
        except Exception as e:
            print(f"⚠️ Error for speaker {speaker}: {e}")
    
    if not s1_sentences and not s2_sentences and not s3_sentences:
        print("❌ No audio generated for any speaker")
        s1 = s2 = s3 = np.zeros(16000)
        sum_path = os.path.join(save_dir, 'sum.wav')
        sf.write(sum_path, s1, 16000)
        return s1, s2, s3, sum_path
    
    s1_combined = torch.concat(s1_sentences, dim=0) if s1_sentences else torch.tensor([])
    s2_combined = torch.concat(s2_sentences, dim=0) if s2_sentences else torch.tensor([])
    s3_combined = torch.concat(s3_sentences, dim=0) if s3_sentences else torch.tensor([])
    
    max_len = max(len(s1_combined), len(s2_combined), len(s3_combined), 1)
    
    if len(s1_combined) == 0:
        s1_combined = torch.zeros(max_len)
    elif len(s1_combined) < max_len:
        s1_combined = torch.cat([s1_combined, torch.zeros(max_len - len(s1_combined))])
    
    if len(s2_combined) == 0:
        s2_combined = torch.zeros(max_len)
    elif len(s2_combined) < max_len:
        s2_combined = torch.cat([s2_combined, torch.zeros(max_len - len(s2_combined))])
    
    if len(s3_combined) == 0:
        s3_combined = torch.zeros(max_len)
    elif len(s3_combined) < max_len:
        s3_combined = torch.cat([s3_combined, torch.zeros(max_len - len(s3_combined))])
    
    sum_combined = s1_combined + s2_combined + s3_combined
    
    save_path1 = f'{save_dir}/s1.wav'
    save_path2 = f'{save_dir}/s2.wav'
    save_path3 = f'{save_dir}/s3.wav'
    save_path_sum = f'{save_dir}/sum.wav'
    
    sf.write(save_path1, s1_combined.numpy(), 24000)
    sf.write(save_path2, s2_combined.numpy(), 24000)
    sf.write(save_path3, s3_combined.numpy(), 24000)
    sf.write(save_path_sum, sum_combined.numpy(), 24000)
    
    s1, _ = librosa.load(save_path1, sr=16000)
    s2, _ = librosa.load(save_path2, sr=16000)
    s3, _ = librosa.load(save_path3, sr=16000)
    
    print(f"✅ 3-speaker audio generated")
    return s1, s2, s3, save_path_sum



def initialize_models():
    """Initialize the WAN models with caching - only loads once!"""
    global wan_pipeline, wav2vec_feature_extractor, audio_encoder, _models_cache, wan, WAN_CONFIGS
    
    # ✅ CHECK: Are models already loaded?
    if _models_cache['initialized']:
        logger.info("⚡ Models already loaded! Using cached models (INSTANT)")
        wan_pipeline = _models_cache['wan_pipeline']
        wav2vec_feature_extractor = _models_cache['wav2vec_feature_extractor']
        audio_encoder = _models_cache['audio_encoder']
        logger.info(f"✅ Globals updated from cache")
        return
    
    try:
        logger.info("🔄 Loading models for the FIRST TIME (this takes 15-20 minutes)...")
        
        # ✅ Import wan HERE - after CUDA is ready
        if wan is None:
            logger.info("📦 Importing WAN modules...")
            import wan as wan_module
            wan = wan_module
            
            # ✅ FIX: Properly import WAN_CONFIGS
            from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS as WAN_CONFIGS_import
            WAN_CONFIGS = WAN_CONFIGS_import  # ← Update global
            logger.info("✅ WAN modules imported successfully")
        
        # Initialize Wav2Vec2
        logger.info("⏳ Loading Wav2Vec2 model...")
        wav2vec_feature_extractor, audio_encoder = custom_init(
            device, 
            '/content/drive/MyDrive/weights/chinese-wav2vec2-base'
        )
        logger.info("✅ Wav2Vec2 loaded")
        
        # Initialize WAN pipeline
        logger.info("⏳ Loading WAN pipeline (this takes 10-15 minutes)...")
        cfg = WAN_CONFIGS["multitalk-14B"]  # ← This uses the global now
        
        wan_pipeline = wan.MultiTalkPipeline(
            config=cfg,
            checkpoint_dir='/content/drive/MyDrive/weights/Wan2.1-I2V-14B-480P',
            quant_dir='/content/drive/MyDrive/weights/MeiGen-MultiTalk',
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
            t5_cpu=False,
            lora_dir=['/content/drive/MyDrive/weights/MeiGen-MultiTalk/quant_models/quant_model_int8_FusionX.safetensors'],
            lora_scales=[1.2],
            quant="int8"
        )
        
        if wan_pipeline is None:
            raise Exception("WAN pipeline creation failed")
        
        # ✅ CACHE THE MODELS
        _models_cache['wan_pipeline'] = wan_pipeline
        _models_cache['wav2vec_feature_extractor'] = wav2vec_feature_extractor
        _models_cache['audio_encoder'] = audio_encoder
        _models_cache['initialized'] = True
        
        logger.info("✅✅ ALL MODELS LOADED AND CACHED!")
        logger.info("🎉 Future requests will be INSTANT (no 20-minute wait)")
        
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise



def generate_video_worker(input_data, output_path, job_id):
    """Worker function with better error handling"""
    try:
        logger.info(f"\n{'='*80}")
        logger.info(f"🎬 VIDEO GENERATION STARTED - Job: {job_id}")
        logger.info(f"{'='*80}")
        
        # Log complete input data
        input_log = {
            "prompt": input_data.get("prompt", "N/A"),
            "cond_image": input_data.get("cond_image", "N/A"),
            "audio_type": input_data.get("audio_type", "N/A"),
            "video_audio": input_data.get("video_audio", "N/A"),
            "speakers": list(input_data.get("cond_audio", {}).keys()),
            "speaker_details": {}
        }
        
        # Add embedding details
        for key, path in input_data.get("cond_audio", {}).items():
            if os.path.exists(path):
                emb = torch.load(path)
                input_log["speaker_details"][key] = {
                    "path": path,
                    "shape": str(emb.shape),
                    "dtype": str(emb.dtype)
                }
        
        log_json("📋 VIDEO GENERATION INPUT DATA", input_log)
        
        
        if wan_pipeline is None:
            error_msg = "❌ wan_pipeline is None! Models not loaded properly."
            logger.error(error_msg)
            raise Exception(error_msg)
            
        logger.info("✅ WAN pipeline verified, starting generation...")
        logger.info(f"📋 Input data for WAN pipeline:")
        logger.info(f"   Prompt: {input_data.get('prompt', 'N/A')}")
        logger.info(f"   Image: {input_data.get('cond_image', 'N/A')}")
        logger.info(f"   Cond audio keys: {list(input_data.get('cond_audio', {}).keys())}")
        logger.info(f"   Video audio: {input_data.get('video_audio', 'N/A')}")
        
        # ✅ Validate audio embeddings
        cond_audio = input_data.get('cond_audio', {})
        if not cond_audio or len(cond_audio) == 0:
            raise Exception("❌ No audio embeddings found! Audio processing failed.")
        
        # ✅ Check if all embedding files exist and are valid
        for key, emb_path in cond_audio.items():
            if not os.path.exists(emb_path):
                raise Exception(f"❌ Embedding file missing: {emb_path}")
            
            # Verify embedding can be loaded
            try:
                emb = torch.load(emb_path)
                logger.info(f"✅ {key} embedding validated: shape {emb.shape}")
            except Exception as e:
                raise Exception(f"❌ Failed to load {key} embedding: {e}")
        
        # Create extra_args object
        class ExtraArgs:
            use_teacache = False
            use_apg = False
            teacache_thresh = 0.2
            apg_momentum = -0.75
            apg_norm_threshold = 55
        
        extra_args = ExtraArgs()
        
        # Generate video
        logger.info("⏳ Generating video (this may take 5-10 minutes)...")
        logger.info(f"📊 Using {len(cond_audio)} speaker(s) for video generation")
        
        video = wan_pipeline.generate(
            input_data,
            size_buckget="multitalk-480",
            motion_frame=25,
            frame_num=81,
            shift=2,
            sampling_steps=8,
            text_guide_scale=1.0,
            audio_guide_scale=2.0,
            seed=42,
            offload_model=True,
            max_frames_num=1000,
            color_correction_strength=1.0,
            extra_args=extra_args
        )
        
        # Save video
        video_audio_path = input_data.get('video_audio', '')
        audio_paths = [video_audio_path] if video_audio_path else []
        
        logger.info(f"💾 Saving video to: {output_path}.mp4")
        save_video_ffmpeg(video, output_path, audio_paths, high_quality_save=False)
        
        logger.info(f"✅ Video generation completed for job {job_id}")
        
    except AssertionError as ae:
        logger.error(f"❌ Assertion Error in video generation: {ae}")
        logger.error("Audio embeddings validation failed")
        import traceback
        logger.error(f"🔍 Full traceback:\n{traceback.format_exc()}")
        error_file = f"{output_path}_error.txt"
        with open(error_file, 'w') as f:
            f.write(f"Job {job_id} failed - Assertion Error:\n{str(ae)}\n\nTraceback:\n{traceback.format_exc()}")
            
    except Exception as e:
        logger.error(f"❌ Error in video generation for job {job_id}: {e}")
        import traceback
        logger.error(f"🔍 Full traceback:\n{traceback.format_exc()}")
        error_file = f"{output_path}_error.txt"
        with open(error_file, 'w') as f:
            f.write(f"Job {job_id} failed:\n{str(e)}\n\nTraceback:\n{traceback.format_exc()}")




@app.route('/')
def home():
    return jsonify({"message": "WAN Video Generation API", "status": "running"})

@app.route('/api/generate-tts-video', methods=['POST'])
def generate_tts_video():
    """Endpoint for TTS-based video generation"""
    try:
        logger.info(f"\n{'='*80}")
        logger.info("📤 TTS VIDEO REQUEST RECEIVED")
        logger.info(f"{'='*80}")
        image_file = request.files.get('image')
        config_data = json.loads(request.form.get('config', '{}'))

        log_json("📨 REQUEST CONFIG", config_data)
        
        if not image_file:
            return jsonify({"error": "No image file provided"}), 400
        
        job_id = str(uuid.uuid4())
        job_folder = os.path.join(UPLOAD_FOLDER, job_id)
        os.makedirs(job_folder, exist_ok=True)
        
        image_path = os.path.join(job_folder, image_file.filename)
        image_file.save(image_path)
        
        input_data = {
            "prompt": config_data.get("prompt", "A new avatar video."),
            "cond_image": image_path,
            "audio_type": "para",
            "tts_audio": config_data.get("tts_audio", {}),
            "cond_audio": {}
        }
        
        logger.info(f"🎯 TTS request - Job: {job_id}")
        logger.info(f"   Image: {image_file.filename}")
        logger.info(f"   Prompt: {input_data['prompt']}")
        logger.info(f"   TTS voices: {list(input_data['tts_audio'].keys())}")
        
        audio_save_dir = os.path.join(job_folder, 'audio')
        os.makedirs(audio_save_dir, exist_ok=True)
        
        tts_audio = input_data['tts_audio']
        text = tts_audio.get('text', '')
        
        logger.info(f"📝 Dialogue text: '{text}'")
        
        # Determine number of speakers and process accordingly
        num_speakers = 0
        if 'human1_voice' in tts_audio and tts_audio['human1_voice']:
            num_speakers += 1
        if 'human2_voice' in tts_audio and tts_audio['human2_voice']:
            num_speakers += 1
        if 'human3_voice' in tts_audio and tts_audio['human3_voice']:
            num_speakers += 1
        
        logger.info(f"🎤 Number of speakers: {num_speakers}")
        
        if num_speakers == 1:
            logger.info("→ Processing single speaker")
            new_human_speech1, sum_audio = process_tts_single(
                text, audio_save_dir, tts_audio['human1_voice']
            )
            audio_embedding_1 = get_embedding(
                new_human_speech1, wav2vec_feature_extractor, audio_encoder, 16000, device
            )
            emb1_path = os.path.join(audio_save_dir, '1.pt')
            torch.save(audio_embedding_1, emb1_path)
            input_data['cond_audio']['person1'] = emb1_path
            input_data['video_audio'] = sum_audio
            
        elif num_speakers == 2:
            logger.info("→ Processing 2 speakers")
            new_human_speech1, new_human_speech2, sum_audio = process_tts_multi(
                text, audio_save_dir,
                tts_audio['human1_voice'],
                tts_audio['human2_voice']
            )
            audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder, 16000, device)
            audio_embedding_2 = get_embedding(new_human_speech2, wav2vec_feature_extractor, audio_encoder, 16000, device)
            
            emb1_path = os.path.join(audio_save_dir, '1.pt')
            emb2_path = os.path.join(audio_save_dir, '2.pt')
            
            torch.save(audio_embedding_1, emb1_path)
            torch.save(audio_embedding_2, emb2_path)
            
            input_data['cond_audio']['person1'] = emb1_path
            input_data['cond_audio']['person2'] = emb2_path
            input_data['video_audio'] = sum_audio
            
        elif num_speakers >= 3:
            logger.info("→ Processing 3 speakers")
            new_human_speech1, new_human_speech2, new_human_speech3, sum_audio = process_tts_triple(
                text, audio_save_dir,
                tts_audio['human1_voice'],
                tts_audio['human2_voice'],
                tts_audio.get('human3_voice', '')
            )
            audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder, 16000, device)
            audio_embedding_2 = get_embedding(new_human_speech2, wav2vec_feature_extractor, audio_encoder, 16000, device)
            audio_embedding_3 = get_embedding(new_human_speech3, wav2vec_feature_extractor, audio_encoder, 16000, device)
            
            emb1_path = os.path.join(audio_save_dir, '1.pt')
            emb2_path = os.path.join(audio_save_dir, '2.pt')
            emb3_path = os.path.join(audio_save_dir, '3.pt')
            
            torch.save(audio_embedding_1, emb1_path)
            torch.save(audio_embedding_2, emb2_path)
            torch.save(audio_embedding_3, emb3_path)
            
            input_data['cond_audio']['person1'] = emb1_path
            input_data['cond_audio']['person2'] = emb2_path
            input_data['cond_audio']['person3'] = emb3_path
            input_data['video_audio'] = sum_audio
        
        logger.info(f"✅ Audio processed - embeddings: {list(input_data['cond_audio'].keys())}")
        
        # Start video generation
        output_path = os.path.join(OUTPUT_FOLDER, f"video_{job_id}")
        thread = threading.Thread(
            target=generate_video_worker,
            args=(input_data, output_path, job_id)
        )
        thread.start()
        
        return jsonify({
            "job_id": job_id,
            "status": "started",
            "message": f"Video generation started for {num_speakers} speaker(s)"
        })
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
    
# Add this helper function for JSON logging
def log_json(label, data):
    """Pretty print JSON data to logger"""
    logger.info(f"\n{'='*70}")
    logger.info(f"📋 {label}")
    logger.info(f"{'='*70}")
    try:
        logger.info(json.dumps(data, indent=2, default=str))
    except Exception as e:
        logger.info(f"[JSON serialization error]: {str(data)}")
    logger.info(f"{'='*70}\n")


@app.route('/api/generate-audio-video', methods=['POST'])
def generate_audio_video():
    """Endpoint for audio file-based video generation"""
    try:
        logger.info(f"\n{'='*80}")
        logger.info("📤 AUDIO VIDEO REQUEST RECEIVED")
        logger.info(f"{'='*80}")
        
        image_file = request.files.get('image')
        audio_files = request.files.getlist('audio_files')
        config_data = json.loads(request.form.get('config', '{}'))
        
        # ✅ ADD THIS
        request_info = {
            "image": image_file.filename if image_file else None,
            "audio_files_count": len(audio_files),
            "audio_files": [f.filename for f in audio_files],
            "config": config_data
        }
        log_json("📨 REQUEST INFO", request_info)
        
        if not image_file or not audio_files:
            return jsonify({"error": "Image and audio files are required"}), 400
        
        job_id = str(uuid.uuid4())
        job_folder = os.path.join(UPLOAD_FOLDER, job_id)
        os.makedirs(job_folder, exist_ok=True)
        
        image_path = os.path.join(job_folder, image_file.filename)
        image_file.save(image_path)
        
        audio_save_dir = os.path.join(job_folder, 'audio')
        os.makedirs(audio_save_dir, exist_ok=True)
        
        num_speakers = len(audio_files)
        logger.info(f"🎯 Audio-to-video request - Job: {job_id}")
        logger.info(f"   Number of speakers: {num_speakers}")
        
        cond_audio = {}
        all_audio_arrays = []
        
        # ✅ Process uploaded audio files
        for i, audio_file in enumerate(audio_files):
            audio_path = os.path.join(job_folder, f'person{i+1}.wav')
            audio_file.save(audio_path)
            logger.info(f"   Audio {i+1}: {audio_file.filename}")
            
            # Process audio
            speech = audio_prepare_single(audio_path)
            all_audio_arrays.append(speech)
            
            # Get embedding
            emb = get_embedding(speech, wav2vec_feature_extractor, audio_encoder, 16000, device)
            emb_path = os.path.join(audio_save_dir, f'{i+1}.pt')
            torch.save(emb, emb_path)
            
            cond_audio[f'person{i+1}'] = emb_path
            logger.info(f"✅ Audio {i+1} processed")
        
        # ✅ CRITICAL: Always provide 3 speakers (pad with zeros if needed)
        logger.info(f"📝 Padding embeddings for {3 - num_speakers} missing speaker(s)...")
        
        # Get reference shape from first embedding
        ref_emb = torch.load(cond_audio['person1'])
        seq_len, batch_size, embed_dim = ref_emb.shape
        
        # Create zero embeddings for missing speakers
        for i in range(num_speakers + 1, 4):  # Fill up to person3
            zero_emb = torch.zeros(seq_len, batch_size, embed_dim)
            emb_path = os.path.join(audio_save_dir, f'{i}.pt')
            torch.save(zero_emb, emb_path)
            cond_audio[f'person{i}'] = emb_path
            logger.info(f"✅ Created zero embedding for person{i}")
        
        # Create mixed audio
        if all_audio_arrays:
            max_len = max([len(a) for a in all_audio_arrays])
            padded = [np.pad(a, (0, max_len - len(a))) for a in all_audio_arrays]
            sum_audio = np.sum(padded, axis=0)
            sum_audio_path = os.path.join(audio_save_dir, 'sum.wav')
            sf.write(sum_audio_path, sum_audio, 16000)
            logger.info(f"✅ Mixed audio created")
        
        # Prepare input data
        input_data = {
            "prompt": config_data.get("prompt", "A person speaking with natural expressions."),
            "cond_image": image_path,
            "audio_type": "para",
            "cond_audio": cond_audio,
            "video_audio": sum_audio_path
        }
        
        logger.info(f"📋 Cond audio keys: {list(cond_audio.keys())}")
        
        # Start video generation
        output_path = os.path.join(OUTPUT_FOLDER, f"video_{job_id}")
        thread = threading.Thread(
            target=generate_video_worker,
            args=(input_data, output_path, job_id)
        )
        thread.start()
        
        return jsonify({
            "job_id": job_id,
            "status": "started",
            "message": f"Video generation started for {num_speakers} speaker(s)"
        })
        
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500

@app.route('/api/status/<job_id>', methods=['GET'])
def get_status(job_id):
    """Check status of a generation job"""
    video_path = os.path.join(OUTPUT_FOLDER, f"video_{job_id}.mp4")
    
    if os.path.exists(video_path):
        return jsonify({
            "job_id": job_id,
            "status": "completed",
            "video_url": f"/api/video/{job_id}"  # Changed to video endpoint
        })
    else:
        # Check if job is still processing
        job_folder = os.path.join(UPLOAD_FOLDER, job_id)
        if os.path.exists(job_folder):
            return jsonify({
                "job_id": job_id,
                "status": "processing"
            })
        else:
            return jsonify({
                "job_id": job_id,
                "status": "not_found"
            }), 404

@app.route('/api/video/<job_id>', methods=['GET'])
def stream_video(job_id):
    """Stream generated video for display in browser"""
    video_path = os.path.join(OUTPUT_FOLDER, f"video_{job_id}.mp4")
    
    if os.path.exists(video_path):
        # Set proper headers for video streaming
        response = send_file(
            video_path,
            as_attachment=False,  # Don't force download
            mimetype='video/mp4',
            conditional=True  # Supports range requests for streaming
        )
        
        # Add CORS headers
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Cross-Origin-Resource-Policy'] = 'cross-origin'
        
        return response
    else:
        return jsonify({"error": "Video not found"}), 404

@app.route('/api/download/<job_id>', methods=['GET'])
def download_video(job_id):
    """Download generated video (for download button)"""
    video_path = os.path.join(OUTPUT_FOLDER, f"video_{job_id}.mp4")
    
    if os.path.exists(video_path):
        return send_file(video_path, as_attachment=True)
    else:
        return jsonify({"error": "Video not found"}), 404
def start_ngrok():
    """Start ngrok tunnel"""
    try:
        # Set your ngrok authtoken here
        # You can get it from https://dashboard.ngrok.com/get-started/your-authtoken
        ngrok.set_auth_token("34emEd0Ln96iv1mA71I65fDQYBd_3gfMq7QJXuH8NEBs7Zmop")
        
        # Start ngrok tunnel
        public_url = ngrok.connect(5000).public_url
        logger.info(f"Ngrok tunnel created: {public_url}")
        return public_url
    except Exception as e:
        logger.error(f"Error starting ngrok: {e}")
        return None

# Update the main block to handle initialization failures
if __name__ == '__main__':
    try:
        # Initialize models
        initialize_models()
        
        # Verify models loaded
        if wan_pipeline is None:
            logger.error("❌ CRITICAL: Models failed to initialize. Cannot start server.")
            exit(1)
            
        # Start ngrok tunnel
        public_url = start_ngrok()
        if public_url:
            logger.info(f"🚀 Server is publicly accessible at: {public_url}")
        
        # Start Flask app
        logger.info("✅ Starting Flask server on port 5000...")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except Exception as e:
        logger.error(f"❌ Failed to start application: {e}")
        exit(1)