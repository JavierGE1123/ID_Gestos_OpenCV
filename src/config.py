# config.py
import json
from dataclasses import dataclass, field
from pathlib import Path

SETTINGS_PATH = Path(__file__).parent / "settings.json"


@dataclass
class KeyBinding:
    key: str
    help: str


@dataclass
class KeyBindings:
    quit: KeyBinding = field(default_factory=lambda: KeyBinding("q", "quit"))
    toggle_mask: KeyBinding = field(default_factory=lambda: KeyBinding("m", "mask"))
    reset_skin: KeyBinding = field(default_factory=lambda: KeyBinding("r", "reset"))
    toggle_contours: KeyBinding = field(default_factory=lambda: KeyBinding("c", "contours"))

    def save(self, path: Path = SETTINGS_PATH):
        """Escribe el archivo que el usuario puede editar."""
        data = {
            name: {"key": b.key, "help": b.help}
            for name, b in vars(self).items()
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path = SETTINGS_PATH) -> "KeyBindings":
        """Carga desde el archivo o crea uno con defaults si no existe."""
        instance = cls()

        if not path.exists():
            instance.save(path)
            print(f"[config] Creado {path} con valores por defecto.")
            return instance

        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"[config] Error leyendo {path}: {e}. Usando defaults.")
            return instance

        for name, binding in vars(instance).items():
            if name in raw and isinstance(raw[name], dict):
                key = raw[name].get("key", binding.key)
                if not isinstance(key, str) or len(key) != 1:
                    print(f"[config] Tecla inválida para '{name}': {key!r}. Usando '{binding.key}'.")
                else:
                    binding.key = key
        return instance

    def help_text(self) -> str:
        return " | ".join(f"{b.key}: {b.help}" for b in vars(self).values())


KEYS = KeyBindings.load()
HELP_TEXT = KEYS.help_text()