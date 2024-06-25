
import codecs

def detect_encoding(file_path):
    encodings = ['utf-8', 'utf-16', 'utf-32', 'latin-1']
    for encoding in encodings:
        try:
            with codecs.open(file_path, 'r', encoding=encoding) as file:
                file.read()
            return encoding
        except UnicodeDecodeError:
            continue
    return None

def convert_to_utf8(file_path):
    detected_encoding = detect_encoding(file_path)
    if detected_encoding and detected_encoding != 'utf-8':
        with codecs.open(file_path, 'r', encoding=detected_encoding) as file:
            content = file.read()
        with codecs.open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        print(f"Converted {file_path} from {detected_encoding} to UTF-8")
    elif detected_encoding == 'utf-8':
        print(f"{file_path} is already in UTF-8 encoding")
    else:
        print(f"Unable to detect encoding for {file_path}")

# Use the function
convert_to_utf8('requirements.txt')