from com_func import resource_path

if __name__ == "__main__":

    file_message = resource_path("assets/message.txt")
    with open(file_message, "r", encoding="utf-8") as file:
        content = file.read()

    print(content)