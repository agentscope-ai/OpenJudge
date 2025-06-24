def detect_consecutive_repetition(text, min_len=3, threshold=10):
    """
    Detect consecutive repeated content in the text.
    :param text: Input text
    :param min_len: Minimum length of the repeated chunk (in words)
    :return: The repeated chunk and its repeat count
    """
    # Split the text into words
    words = text.split()
    # Try different chunk lengths from min_len up to min_len+5
    for n in range(min_len, min(min_len + 5, len(words) // 2 + 1)):
        # Slide a window of size n*2 over the words
        for i in range(len(words) - n * 2 + 1):
            chunk = words[i : i + n]
            next_chunk = words[i + n : i + 2 * n]
            # Check if the current chunk is repeated immediately after itself
            if chunk == next_chunk:
                # Count how many times the chunk is repeated consecutively
                count = 2
                while (
                    i + (count + 1) * n <= len(words)
                    and words[i : i + n] == words[i + n * count : i + n * (count + 1)]
                ):
                    count += 1
                if count > threshold:
                    phrase = " ".join(chunk)
                    return {phrase: count}

    return {}
