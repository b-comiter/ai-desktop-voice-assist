import json
import os


class ContextManager:
    def __init__(self, file_path="data/context.json"):
        self.file_path = file_path
        self._data = self._load_file()

    def _load_file(self):
        """Load JSON file or create empty dict if missing/invalid."""
        if not os.path.exists(self.file_path):
            self._save_file({})
            return {}
        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except json.JSONDecodeError:
            return {}

    def _save_file(self, data):
        """Save the current dictionary to the JSON file."""
        # Ensure the directory exists before saving
        dir_path = os.path.dirname(self.file_path)
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        with open(self.file_path, "w") as f:
            json.dump(data, f, indent=2)

    @property
    def data(self):
        """Access the conversation dictionary."""
        return self._data

    def add_message(self, user_id, role, content):
        """Add a message for a given user_id and auto-save."""
        if user_id not in self._data:
            self._data[user_id] = []
        self._data[user_id].append({"role": role, "content": content})
        self._save_file(self._data)

    def is_user(self, user_id):
        """Check if user_id has a conversation"""
        if user_id in self._data:
            return True
        return False

    def get_history(self, user_id):
        """Get conversation history for a user."""
        self._data = self._load_file()  # Reload to ensure we have the latest data
        return self._data.get(user_id, [])

    def clear_user(self, user_id):
        """Clear conversation history for a specific user and auto-save."""
        if user_id in self._data:
            self._data[user_id] = []
            self._save_file(self._data)
