from typing import List, Tuple, Optional, Dict, Any


class TokenCache:
    """Simple cache to store tokenization results"""
    def __init__(self, max_size: int = 100):
        self.cache: Dict[str, List[Tuple[int, str, float]]] = {}
        self.max_size = max_size
        self.access_order: List[str] = []

    def get(self, key: str) -> Optional[List[Tuple[int, str, float]]]:
        if key in self.cache:
            # Move to end to mark as recently used
            try:
                self.access_order.remove(key)
            except ValueError:
                pass
            self.access_order.append(key)
            return self.cache[key]
        return None

    def set(self, key: str, value: List[Tuple[int, str, float]]):
        if key in self.cache:
            self.cache[key] = value
            try:
                self.access_order.remove(key)
            except ValueError:
                pass
            self.access_order.append(key)
        else:
            if len(self.cache) >= self.max_size:
                # Remove least recently used item
                lru_key = self.access_order.pop(0)
                del self.cache[lru_key]

            self.cache[key] = value
            self.access_order.append(key)

    def clear(self):
        self.cache.clear()
        self.access_order.clear()