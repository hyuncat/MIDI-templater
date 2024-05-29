from pytube import YouTube
import os

def download_YT_video(url, download_path):
    """
    Function to download video from a YouTube video
    @param:
        - url: URL of the YouTube video
        - download_path: Path to download the video file
    @return:
        - full_path: Path to the downloaded video file
    """
    yt = YouTube(url)
    stream = yt.streams.first()
    filename = stream.default_filename.rsplit('.', 1)[0] + "_video." + stream.default_filename.rsplit('.', 1)[1]
    full_path = os.path.join(download_path, filename)

    # delete file if already exists
    if os.path.exists(full_path):
        os.remove(full_path)

    stream.download(download_path, filename=filename)

    print(f"Downloaded {yt.title} (video) to {download_path}")

    return full_path
    

def download_YT_audio(url, download_path):
    """
    Function to download audio from a YouTube video
    @param:
        - url: URL of the YouTube video
        - download_path: Path to download the audio file
    @return:
        - full_path: Path to the downloaded audio file
    """
    yt = YouTube(url)
    stream = yt.streams.filter(only_audio=True).first()
    filename = stream.default_filename.rsplit('.', 1)[0] + "_audio." + stream.default_filename.rsplit('.', 1)[1]
    full_path = os.path.join(download_path, filename)

    # delete file if already exists
    if os.path.exists(full_path):
        os.remove(full_path)

    stream.download(download_path, filename=filename)

    print(f"Downloaded {yt.title} (audio) to {download_path}")

    return full_path


if __name__ == "__main__":
    # test with dayoon's bach fugue
    url = "https://www.youtube.com/watch?v=QMST8I9NBIU"
    download_path = "downloads"
    download_YT_video(url, download_path)
    download_YT_audio(url, download_path)