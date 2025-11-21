#class data getter
import pathlib

class DataGetter:
    """A creative class to fetch file data by name from a fixed base path."""
    
    def __init__(self):
        self.base_path = r"C:\Users\A S P I R E\OneDrive\Documents\Test"  # Dari ibutang a base path kung asa nato kwaon ang data file
    
    def read_file(self, file_name):
        """Reads the file content."""
        try:
            return (pathlib.Path(self.base_path) / file_name).read_text()
        except FileNotFoundError:
            return f"Error: '{file_name}' not found in '{self.base_path}'."
        except Exception as e:
            return f"Error: {e}"
#Usage
getter = DataGetter()
file_name = input("File name: ")  # e.g., "test.txt"
print(getter.read_file(file_name))
