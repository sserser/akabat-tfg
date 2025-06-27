import os


class ConsoleViewer:
    def __init__(self) -> None:
        pass

    def display_menu(self, menu: dict[str, str]) -> None:
        print("MENU OPTIONS:")
        for i, option in enumerate(menu.values()):
            print(f" {i:2d}. {option}")

    def ask_menu_option(
        self, menu: dict[str, str], prompt: str = "Select an option number: "
    ) -> int:
        option_id = -1
        while option_id < 0 or option_id >= len(menu):
            try:
                self.display_menu(menu)
                user_input = input(prompt)
                option_id = int(user_input)
            except:
                self.display_non_valid_option(menu)
        return option_id

    def ask_folder_path(self, prompt: str = "Enter the folder path: "):
        valid_folder = False
        folder_path = None
        while not valid_folder:
            folder_path = self.ask_path(prompt)
            valid_folder = os.path.isdir(folder_path)
        return folder_path

    def ask_file_path(self, prompt: str = "Enter the file path: "):
        valid_file = False
        file_path = None
        while not valid_file:
            file_path = self.ask_path(prompt)
            valid_file = os.path.isfile(file_path)
        return file_path

    def ask_path(self, prompt: str = "Enter the path: ") -> str:
        return self.validate_user_input(prompt)

    def ask_keyword(self, prompt: str = "Enter the keyword: ") -> str:
        return self.validate_user_input(prompt)

    def ask_number_clusters(
        self,
        prompt: str = "Number of keywords groups (0 to apply distance threshold): ",
    ) -> int:
        return self.validate_int_user_input(prompt)

    def ask_distance_threshold(self, prompt: str = "Distance threshold: ") -> int:
        return self.validate_float_user_input(prompt)

    def ask_keyword_groups(
        self,
        prompt: str = "Enter the groups separated by a comma. Example: federated learning, machine learning\nGroups to merge: ",
    ) -> list[str]:
        user_input = self.validate_user_input(prompt)
        if user_input:
            return [group.strip() for group in user_input.lower().split(",")]
        return None

    def validate_int_user_input(self, prompt: str) -> int:
        user_number = None
        while user_number is None:
            try:
                user_input = self.validate_user_input(prompt)
                user_number = int(user_input)
            except:
                if not user_input:  # [exit]
                    break
        return user_number

    def validate_float_user_input(self, prompt: str) -> float:
        user_number = None
        while user_number is None:
            try:
                user_input = self.validate_user_input(prompt)
                user_number = float(user_input)
            except:
                if not user_input:  # [exit]
                    break
        return user_number

    def validate_user_input(self, prompt: str) -> str:
        user_input = input(prompt)
        if user_input != "[exit]":
            return user_input
        return None

    def display_non_valid_option(self, menu: dict[str, str]) -> None:
        print(f"Please, choose a valid option ID from 0 to {len(menu) - 1}.")
