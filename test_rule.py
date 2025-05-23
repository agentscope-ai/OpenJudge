from gallery.rm.code.rm import FormatRM, LengtRM

length_parser = LengtRM(
    name="length"
)

format_parser = FormatRM(
    name="format"
)


if __name__ == "__main__":

    result = length_parser._run(actual_output="这是一条很长很长很长很长很长很长很长很长很长很长很长很长很长很长的答案")
    print(result)

    result = format_parser._run(actual_output="# Intro\n这是intro\n# Body\n这是body\n#Conclusion\n这是conclusion\n")
    print(result)

