"""Configuration management with .ini files"""

import configparser
from pathlib import Path
from typing import Any, Dict


class ConfigManager:
    """Manage configuration from .ini file"""
    
    def __init__(self, config_path: str = "config.ini"):
        self.config_path = Path(config_path)
        self.parser = configparser.ConfigParser()
        self.config_dict = {}
        
        if self.config_path.exists():
            self.parser.read(self.config_path)
            self._load_config()
        else:
            raise FileNotFoundError(f"Config file not found: {config_path}")
    
    def _load_config(self):
        """Load configuration into dictionary"""
        for section in self.parser.sections():
            self.config_dict[section] = dict(self.parser.items(section))
    
    def get(self, section: str, key: str, default: Any = None) -> Any:
        """
        Get configuration value with type conversion
        
        Parameters
        ----------
        section : str
            Configuration section
        key : str
            Configuration key
        default : Any
            Default value if key not found
        
        Returns
        -------
        Any
            Configuration value with appropriate type
        """
        try:
            value = self.parser.get(section, key)
            
            # Type conversion
            if value.lower() in ('true', 'false'):
                return value.lower() == 'true'
            elif value.isdigit():
                return int(value)
            else:
                try:
                    return float(value)
                except ValueError:
                    return value
        except (configparser.NoSectionError, configparser.NoOptionError):
            return default
    
    def set(self, section: str, key: str, value: Any):
        """Set configuration value"""
        if not self.parser.has_section(section):
            self.parser.add_section(section)
        self.parser.set(section, key, str(value))
        self.config_dict[section][key] = value
    
    def save(self):
        """Save configuration to file"""
        with open(self.config_path, 'w') as f:
            self.parser.write(f)