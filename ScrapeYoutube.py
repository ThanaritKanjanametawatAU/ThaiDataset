import os
import re
import json
from pytube import Channel, YouTube
from moviepy.audio.io.AudioFileClip import AudioFileClip
from pydub import AudioSegment
from pydub.silence import split_on_silence

def get_channel_id(url):
    """Extract channel ID from various YouTube channel URL formats."""
    try:
        if '@' in url:
            return url.split('@')[1]
        elif 'channel/' in url:
            return url.split('channel/')[1].split('/')[0]
        elif 'c/' in url:
            return url.split('c/')[1].split('/')[0]
        elif 'user/' in url:  # Legacy username format (less reliable now)
            return url.split('user/')[1].split('/')[0]  # This might not work reliably anymore
        else:
            raise ValueError("Invalid YouTube channel URL format")
    except IndexError:
        raise ValueError("Invalid YouTube channel URL format")
    except Exception as e:
        raise ValueError(f"Error parsing channel URL: {str(e)}")


def sanitize_filename(title):
    """Remove invalid characters from filename."""
    return re.sub(r'[<>:"/\\|?*]', '', title)

def split_audio(mp3_path, output_dir, min_length_ms=1000, max_length_ms=10000):
    """Split audio file into chunks based on silence detection."""
    try:
        audio = AudioSegment.from_mp3(mp3_path)
        chunks = split_on_silence(
            audio,
            min_silence_len=500,
            silence_thresh=-40,
            keep_silence=100
        )

        processed_chunks = []
        current_chunk = AudioSegment.empty()

        for chunk in chunks:
            if len(current_chunk) + len(chunk) < max_length_ms:
                current_chunk += chunk
            else:
                if len(current_chunk) >= min_length_ms:
                    processed_chunks.append(current_chunk)
                current_chunk = chunk

        if len(current_chunk) >= min_length_ms:
            processed_chunks.append(current_chunk)

        metadata = []
        base_filename = os.path.splitext(os.path.basename(mp3_path))[0]

        for i, chunk in enumerate(processed_chunks):
            chunk_path = os.path.join(output_dir, f"{base_filename}_{i+1:04d}.mp3")
            chunk.export(chunk_path, format="mp3")

            metadata.append({
                "file": f"{base_filename}_{i+1:04d}.mp3",
                "duration": len(chunk) / 1000,
                "start_time": sum(len(c) for c in processed_chunks[:i]) / 1000
            })

        metadata_path = os.path.join(output_dir, f"{base_filename}_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)

        return len(processed_chunks)

    except Exception as e:
        print(f"Error splitting audio {mp3_path}: {str(e)}")
        return 0

def download_audio(url, output_dir):
    """Download a single video and convert to MP3."""
    try:
        yt = YouTube(url)  # Removed use_oauth; pytube handles it automatically if needed

        video_title = sanitize_filename(yt.title)
        print(f"Processing: {video_title}")

        audio_stream = yt.streams.filter(only_audio=True).first()
        if not audio_stream:
            print(f"No audio stream found for {video_title}")
            return False

        mp4_path = os.path.join(output_dir, f"{video_title}.mp4")  # Download as MP4 first
        audio_stream.download(output_path=output_dir, filename=f"{video_title}.mp4")

        mp3_path = os.path.join(output_dir, f"{video_title}.mp3")
        audio_clip = AudioFileClip(mp4_path)
        audio_clip.write_audiofile(mp3_path)
        audio_clip.close()

        os.remove(mp4_path)  # Remove the MP4 after conversion

        segments_dir = os.path.join(output_dir, f"{video_title}_segments")
        if not os.path.exists(segments_dir):
            os.makedirs(segments_dir)

        num_segments = split_audio(mp3_path, segments_dir)
        print(f"Created {num_segments} segments")

        os.remove(mp3_path)

        print(f"Successfully processed: {video_title}")
        return True

    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False


def download_channel(channel_url, output_dir="downloads"):
  try:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        channel_id = get_channel_id(channel_url)
        print(f"Attempting to download from channel: @{channel_id}")

        channel = Channel(channel_url) # directly use the provided url
        channel_dir = os.path.join(output_dir, sanitize_filename(channel.channel_name or channel_id)) #fallback to channel id if name is not available
        if not os.path.exists(channel_dir):
            os.makedirs(channel_dir)

        successful = 0
        failed = 0

        for video_url in channel.video_urls:
            if download_audio(video_url, channel_dir):
                successful += 1
            else:
                failed += 1

        print(f"\nDownload complete!")
        print(f"Successfully processed: {successful} videos")
        print(f"Failed downloads: {failed} videos")

  except Exception as e:
        print(f"Error processing channel: {str(e)}")
        print("Please make sure the channel URL is correct and the channel is accessible.")



if __name__ == "__main__":
    print("YouTube Channel Audio Downloader and Splitter")
    print("--------------------------------------------")

    channel_url = input("Enter the YouTube channel URL: ")

    output_dir = input("Enter output directory (press Enter for default 'downloads'): ")
    if not output_dir:
        output_dir = "downloads"

    download_channel(channel_url, output_dir)