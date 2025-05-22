from src.rm.parser import FormatParser, LengthParser

length_parser = LengthParser(
    name="length"
)

format_parser = FormatParser(
    name="format"
)


if __name__ == "__main__":

    result = length_parser._run(actual_output="这是一条很长很长很长很长很长很长很长很长很长很长很长很长很长很长的答案")
    print(result)

    result = format_parser._run(actual_output="# Intro\n这是intro\n# Body\n这是body\n#Conclusion\n这是conclusion\n")
    print(result)

