import os

def replace_in_file(file_path, old_text, new_text):
    """
    Replaces all occurrences of old_text with new_text in the specified file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Replace the text
    updated_content = content.replace(old_text, new_text)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(updated_content)

def replace_in_project(directory, old_text, new_text):
    """
    Recursively replaces all occurrences of old_text with new_text in all files in the directory.
    """
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):  # Only process Python files
                file_path = os.path.join(root, file)
                replace_in_file(file_path, old_text, new_text)
                print(f"Updated: {file_path}")


if __name__ == "__main__":

    # # Replace 'flag_setting_init' with 'FLAG_SETTING_INIT' in the specified file
    # file_path = r'D:\Workspace\app\src\voice_processing\voice_output.py'
    # file_path = os.path.join(os.getcwd(), "app", "src", "voice_processing", "voice_output.py")
    # print(f"Path to app directory: {file_path}")
    # replace_in_file(file_path, 'flag_setting_init', 'FLAG_SETTING_INIT')

    # Replace 'flag_setting_init' with 'FLAG_SETTING_INIT' in the project directory
    project_directory = os.path.join(os.getcwd(), "app", "src")
    print(f"Path to app directory: {project_directory}")
    replace_in_project(project_directory, 'FLAG_STATUSBAR', 'FLAG_STATUSBAR123')
