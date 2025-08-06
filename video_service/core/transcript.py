# 스크립트 추출 기능 (API, Whisper 모두)
import os
import re
from youtube_transcript_api import YouTubeTranscriptApi
import whisper
import yt_dlp
import torch
from faster_whisper import WhisperModel


# 유튜브 영상 URL에서 video_id 추출 함수
def _extract_video_id(url: str) -> str:
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if match:
        return match.group(1)
    raise ValueError("유효한 유튜브 URL에서 Video ID를 찾을 수 없습니다.")


# 유튜브 영상 제목을 가져오는 함수
def get_youtube_title(url: str) -> str:
    print("--- 영상 제목 추출 (yt-dlp) ---")
    try:
        ydl_opts = {'quiet': True} # 로그를 조용히 처리
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            title = info.get('title', '요리명을 추출할 수 없습니다.')
            print(f"✅ 유튜브 영상 제목: {title}")
            return title
        
    except Exception as e:
        print(f"❌ ERROR: 영상 제목 가져오기 실패: {e}")
        return "요리명을 추출할 수 없습니다."


# youtube-transcript-api으로 자막 가져오기 (한국어 우선 시도, 없으면 영어로 시도)
def _get_transcript_from_api(video_id: str) -> str:
    yta = YouTubeTranscriptApi()
    transcript_list = yta.fetch(video_id, languages=['ko', 'en'])
    return " ".join([d.text for d in transcript_list])


# 자막이 없는 경우 Whisper 사용
def _get_transcript_from_audio(url: str) -> str:
    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(temp_dir, '%(id)s.%(ext)s'),
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }],
        'quiet': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            video_id = info['id']
            ext = info['ext']
            audio_file = os.path.join(temp_dir, f"{video_id}.m4a")
            if not os.path.exists(audio_file):
                audio_file = os.path.join(temp_dir, f"{video_id}.{ext}")
                if not os.path.exists(audio_file):
                    raise FileNotFoundError("다운로드된 오디오 파일을 찾을 수 없습니다.")

        print(f"✅ 오디오 다운로드 완료: {audio_file}")

        # faster-whisper 모델 로드 및 음성 인식
        print("🎤 Faster-Whisper 음성 인식 시작...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        model_size = "medium"  # 필요에 따라 tiny, base, small, medium, large 등 선택

        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Faster-Whisper device:", device)

        segments, info = model.transcribe(audio_file, language="ko")
        transcript_text = " ".join([segment.text for segment in segments])
        print(f"✅ Faster-Whisper 음성 인식 완료: {transcript_text[:100]}...")

    finally:
        if 'audio_file' in locals() and os.path.exists(audio_file):
            os.remove(audio_file)
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
            
    return transcript_text


# 유튜브 스크립트를 가져오는 메인 함수.
# API 방식을 먼저 시도하고, 실패 시 Whisper 방식을 사용합니다.
def get_youtube_transcript(url: str, use_whisper_only: bool = False) -> str:
    video_id = _extract_video_id(url)
    
    if not use_whisper_only:
        try:
            print("INFO: 1차 시도 - 자막 API를 통해 스크립트 추출을 시작합니다.")
            return _get_transcript_from_api(video_id)
        except Exception as e:
            print(f"INFO: 자막 API 사용 불가 ({e}). \n 2차 시도 - Whisper 음성 인식을 시작합니다.")

    # 1차 시도 실패 또는 Whisper만 사용하도록 설정된 경우
    try:
        return _get_transcript_from_audio(url)
    except Exception as e:
        print(f"ERROR: 모든 스크립트 추출 방법에 실패했습니다: {e}")
        raise


# 영상 길이를 초 단위로 반환하는 함수 (yt-dlp 사용)
def get_youtube_duration(url: str) -> int:
    try:
        ydl_opts = {'quiet': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            duration = info.get('duration', 0)  # 초 단위
            return duration
    except Exception as e:
        print(f"ERROR: 영상 길이 추출 실패: {e}")
        return 0