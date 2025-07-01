from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter # Import a formatter
import re

def get_youtube_video_id(url):
    """Extracts the YouTube video ID from a URL."""
    video_id_match = re.search(
        r"(?:youtube\.com\/(?:watch\?v=|embed\/|v\/)|youtu\.be\/)([a-zA-Z0-9_-]{11})",
        url
    )
    if video_id_match:
        return video_id_match.group(1)
    return None

def fetch_youtube_transcript(video_url):
    """
    Fetches the transcript for a given YouTube video URL.
    Returns the full transcript text.
    """
    video_id = get_youtube_video_id(video_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL provided.")

    try:
        # Try to get the transcript in English
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(['en', 'en-US'])

        # Use a formatter to get plain text
        formatter = TextFormatter()
        # The format_transcript method returns the formatted text directly
        full_transcript_text = formatter.format_transcript(transcript.fetch())
        
        # We don't strictly need the raw data for our RAG system currently,
        # just the combined text. If you need the structured data, we'd need to handle it differently.
        # For now, let's just return the text.
        return full_transcript_text, transcript.fetch() # Still return fetch() in case needed

    except Exception as e:
        # If English transcripts are not found or fail, try a broader set
        print(f"English transcript not found or failed for video ID {video_id}: {e}. Trying other languages...")
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            language_codes = ['en', 'en-US', 'a.en', 'de', 'es', 'fr', 'it', 'ja', 'ko', 'pt', 'ru', 'zh-Hans', 'zh-Hant']
            
            transcript = None
            for lang in language_codes:
                try:
                    transcript = transcript_list.find_transcript([lang])
                    print(f"Found transcript for language: {lang}")
                    # Use a formatter to get plain text
                    formatter = TextFormatter()
                    full_transcript_text = formatter.format_transcript(transcript.fetch())
                    return full_transcript_text, transcript.fetch()
                except Exception as lang_exception:
                    # Print the specific language failure to debug if needed
                    # print(f"Failed to get transcript for {lang}: {lang_exception}")
                    continue # Try next language if this one fails

            raise RuntimeError("No suitable transcript found in the specified languages.")
                
        except Exception as e_fallback:
            raise RuntimeError(f"Could not fetch transcript for video: {e_fallback}")
