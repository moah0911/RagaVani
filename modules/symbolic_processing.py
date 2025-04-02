"""
Symbolic Processing Module for RagaVani

This module provides symbolic processing for Indian classical music.
It includes classes for raga grammar, tala patterns, and composition analysis.
"""

import json
import os
import logging
import random
from typing import Dict, List, Any, Optional, Tuple, Union

# Configure logging
logger = logging.getLogger(__name__)

# Define paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAGA_GRAMMAR_PATH = os.path.join(DATA_DIR, "raga_grammar.json")
TALA_PATTERNS_PATH = os.path.join(DATA_DIR, "tala_patterns.json")

class RagaGrammar:
    """
    Class for handling raga grammar rules
    """
    
    def __init__(self, grammar_path: Optional[str] = None):
        """
        Initialize the raga grammar
        
        Parameters:
            grammar_path (str, optional): Path to the grammar JSON file
        """
        self.grammar_path = grammar_path or RAGA_GRAMMAR_PATH
        self.grammar = self._load_grammar()
        
    def _load_grammar(self) -> Dict[str, Any]:
        """
        Load the raga grammar from JSON file
        
        Returns:
            dict: Raga grammar
        """
        try:
            if os.path.exists(self.grammar_path):
                with open(self.grammar_path, 'r') as f:
                    grammar = json.load(f)
                logger.info(f"Loaded raga grammar from {self.grammar_path}")
                return grammar
            else:
                logger.warning(f"Raga grammar file not found at {self.grammar_path}")
                return {"ragas": {}}
        except Exception as e:
            logger.error(f"Error loading raga grammar: {str(e)}")
            return {"ragas": {}}
    
    def get_raga_rules(self, raga_name: str) -> Dict[str, Any]:
        """
        Get grammar rules for a specific raga
        
        Parameters:
            raga_name (str): Name of the raga
            
        Returns:
            dict: Raga grammar rules
        """
        if not self.grammar or "ragas" not in self.grammar:
            return {}
            
        ragas = self.grammar["ragas"]
        if raga_name in ragas:
            return ragas[raga_name]
        else:
            logger.warning(f"Raga '{raga_name}' not found in grammar")
            return {}
    
    def get_all_ragas(self) -> List[str]:
        """
        Get a list of all ragas in the grammar
        
        Returns:
            list: List of raga names
        """
        if not self.grammar or "ragas" not in self.grammar:
            return []
            
        return list(self.grammar["ragas"].keys())
    
    def validate_phrase(self, phrase: str, raga_name: str) -> Dict[str, Any]:
        """
        Validate a phrase against raga grammar rules
        
        Parameters:
            phrase (str): Phrase to validate (space-separated swaras)
            raga_name (str): Name of the raga
            
        Returns:
            dict: Validation results
        """
        raga_rules = self.get_raga_rules(raga_name)
        if not raga_rules:
            return {"valid": False, "errors": [f"Raga '{raga_name}' not found in grammar"]}
        
        # Parse the phrase
        swaras = phrase.strip().split()
        if not swaras:
            return {"valid": False, "errors": ["Empty phrase"]}
        
        errors = []
        warnings = []
        
        # Check for forbidden phrases
        if "forbidden_phrases" in raga_rules:
            for forbidden in raga_rules["forbidden_phrases"]:
                forbidden_swaras = forbidden.split()
                for i in range(len(swaras) - len(forbidden_swaras) + 1):
                    if swaras[i:i+len(forbidden_swaras)] == forbidden_swaras:
                        errors.append(f"Forbidden phrase '{forbidden}' found at position {i}")
        
        # Check for allowed swaras
        if "aroha" in raga_rules and "avaroha" in raga_rules:
            allowed_swaras = set(raga_rules["aroha"] + raga_rules["avaroha"])
            for i, swara in enumerate(swaras):
                if swara not in allowed_swaras:
                    errors.append(f"Swara '{swara}' at position {i} is not allowed in raga '{raga_name}'")
        
        # Check for characteristic phrases
        if "characteristic_phrases" in raga_rules:
            found_characteristic = False
            for char_phrase in raga_rules["characteristic_phrases"]:
                char_swaras = char_phrase.split()
                for i in range(len(swaras) - len(char_swaras) + 1):
                    if swaras[i:i+len(char_swaras)] == char_swaras:
                        found_characteristic = True
                        break
                if found_characteristic:
                    break
            
            if not found_characteristic:
                warnings.append(f"No characteristic phrase of raga '{raga_name}' found")
        
        # Check for vadi and samvadi
        if "vadi" in raga_rules and "samvadi" in raga_rules:
            vadi = raga_rules["vadi"]
            samvadi = raga_rules["samvadi"]
            
            if vadi not in swaras:
                warnings.append(f"Vadi swara '{vadi}' not found in phrase")
            
            if samvadi not in swaras:
                warnings.append(f"Samvadi swara '{samvadi}' not found in phrase")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def generate_phrase(self, raga_name: str, length: int = 8) -> str:
        """
        Generate a phrase following raga grammar rules
        
        Parameters:
            raga_name (str): Name of the raga
            length (int): Length of the phrase in swaras
            
        Returns:
            str: Generated phrase
        """
        raga_rules = self.get_raga_rules(raga_name)
        if not raga_rules:
            return ""
        
        # Get allowed phrases
        allowed_phrases = raga_rules.get("allowed_phrases", [])
        if not allowed_phrases and "chalan" in raga_rules:
            allowed_phrases = raga_rules["chalan"]
        
        if not allowed_phrases:
            # Fall back to aroha and avaroha
            if "aroha" in raga_rules and "avaroha" in raga_rules:
                aroha = " ".join(raga_rules["aroha"])
                avaroha = " ".join(raga_rules["avaroha"])
                allowed_phrases = [aroha, avaroha]
            else:
                return ""
        
        # Generate phrase by combining allowed phrases
        phrase = []
        while len(phrase) < length:
            # Select a random allowed phrase
            selected_phrase = random.choice(allowed_phrases)
            selected_swaras = selected_phrase.split()
            
            # Add to the phrase
            phrase.extend(selected_swaras)
        
        # Trim to desired length
        phrase = phrase[:length]
        
        return " ".join(phrase)

class TalaPatterns:
    """
    Class for handling tala patterns
    """
    
    def __init__(self, patterns_path: Optional[str] = None):
        """
        Initialize the tala patterns
        
        Parameters:
            patterns_path (str, optional): Path to the patterns JSON file
        """
        self.patterns_path = patterns_path or TALA_PATTERNS_PATH
        self.patterns = self._load_patterns()
        
        # Define default patterns if file not found
        if not self.patterns or "talas" not in self.patterns:
            self.patterns = {
                "talas": {
                    "Teentaal": {
                        "beats": 16,
                        "vibhags": [4, 4, 4, 4],
                        "clap_pattern": "X 2 0 3",
                        "bols": "Dha Dhin Dhin Dha | Dha Dhin Dhin Dha | Dha Tin Tin Ta | Ta Dhin Dhin Dha"
                    },
                    "Ektaal": {
                        "beats": 12,
                        "vibhags": [2, 2, 2, 2, 2, 2],
                        "clap_pattern": "X 0 2 0 3 0",
                        "bols": "Dhin Dhin | Dha Dha | Tu Na | Ka Ta | Dhin Dhin | Dha Dha"
                    },
                    "Jhaptaal": {
                        "beats": 10,
                        "vibhags": [2, 3, 2, 3],
                        "clap_pattern": "X 2 0 3",
                        "bols": "Dhin Na | Dhin Dhin Na | Tin Na | Dhin Dhin Na"
                    },
                    "Keherwa": {
                        "beats": 8,
                        "vibhags": [4, 4],
                        "clap_pattern": "X 0",
                        "bols": "Dha Ge Na Ti | Na Ka Dhi Na"
                    },
                    "Rupak": {
                        "beats": 7,
                        "vibhags": [3, 2, 2],
                        "clap_pattern": "0 X 0",
                        "bols": "Tin Tin Na | Dhin Na | Dhin Na"
                    },
                    "Dadra": {
                        "beats": 6,
                        "vibhags": [3, 3],
                        "clap_pattern": "X 0",
                        "bols": "Dha Dhin Na | Ta Tin Na"
                    }
                }
            }
    
    def _load_patterns(self) -> Dict[str, Any]:
        """
        Load the tala patterns from JSON file
        
        Returns:
            dict: Tala patterns
        """
        try:
            if os.path.exists(self.patterns_path):
                with open(self.patterns_path, 'r') as f:
                    patterns = json.load(f)
                logger.info(f"Loaded tala patterns from {self.patterns_path}")
                return patterns
            else:
                logger.warning(f"Tala patterns file not found at {self.patterns_path}")
                return {}
        except Exception as e:
            logger.error(f"Error loading tala patterns: {str(e)}")
            return {}
    
    def get_tala_pattern(self, tala_name: str) -> Dict[str, Any]:
        """
        Get pattern for a specific tala
        
        Parameters:
            tala_name (str): Name of the tala
            
        Returns:
            dict: Tala pattern
        """
        if not self.patterns or "talas" not in self.patterns:
            return {}
            
        talas = self.patterns["talas"]
        if tala_name in talas:
            return talas[tala_name]
        else:
            logger.warning(f"Tala '{tala_name}' not found in patterns")
            return {}
    
    def get_all_talas(self) -> List[str]:
        """
        Get a list of all talas in the patterns
        
        Returns:
            list: List of tala names
        """
        if not self.patterns or "talas" not in self.patterns:
            return []
            
        return list(self.patterns["talas"].keys())
    
    def get_clap_pattern(self, tala_name: str) -> str:
        """
        Get clap pattern for a specific tala
        
        Parameters:
            tala_name (str): Name of the tala
            
        Returns:
            str: Clap pattern
        """
        tala = self.get_tala_pattern(tala_name)
        if not tala:
            return ""
            
        return tala.get("clap_pattern", "")
    
    def get_bols(self, tala_name: str) -> str:
        """
        Get bols for a specific tala
        
        Parameters:
            tala_name (str): Name of the tala
            
        Returns:
            str: Bols
        """
        tala = self.get_tala_pattern(tala_name)
        if not tala:
            return ""
            
        return tala.get("bols", "")
    
    def generate_rhythm_pattern(self, tala_name: str, variations: int = 0) -> str:
        """
        Generate a rhythm pattern for a specific tala
        
        Parameters:
            tala_name (str): Name of the tala
            variations (int): Number of variations to introduce
            
        Returns:
            str: Generated rhythm pattern
        """
        tala = self.get_tala_pattern(tala_name)
        if not tala:
            return ""
        
        # Get the standard bols
        bols = tala.get("bols", "")
        if not bols:
            return ""
        
        # If no variations requested, return standard bols
        if variations <= 0:
            return bols
        
        # Parse the bols
        sections = bols.split("|")
        sections = [section.strip() for section in sections]
        
        # Apply variations
        for _ in range(variations):
            # Select a random section to vary
            section_idx = random.randint(0, len(sections) - 1)
            section = sections[section_idx]
            
            # Split into individual bols
            section_bols = section.split()
            
            # Apply a random variation
            variation_type = random.choice(["substitute", "repeat", "combine"])
            
            if variation_type == "substitute" and len(section_bols) > 0:
                # Substitute a bol
                bol_idx = random.randint(0, len(section_bols) - 1)
                alternatives = ["Dha", "Dhin", "Tin", "Ta", "Na", "Ge", "Ka", "Tu"]
                alternatives = [b for b in alternatives if b != section_bols[bol_idx]]
                section_bols[bol_idx] = random.choice(alternatives)
                
            elif variation_type == "repeat" and len(section_bols) > 1:
                # Repeat a bol
                bol_idx = random.randint(0, len(section_bols) - 2)
                section_bols[bol_idx + 1] = section_bols[bol_idx]
                
            elif variation_type == "combine" and len(section_bols) > 1:
                # Combine two bols
                bol_idx = random.randint(0, len(section_bols) - 2)
                combined = section_bols[bol_idx] + section_bols[bol_idx + 1]
                section_bols = section_bols[:bol_idx] + [combined] + section_bols[bol_idx+2:]
            
            # Update the section
            sections[section_idx] = " ".join(section_bols)
        
        # Recombine the sections
        return " | ".join(sections)

class SymbolicProcessor:
    """
    Class for symbolic processing of Indian classical music
    """
    
    def __init__(self, grammar_path: Optional[str] = None, patterns_path: Optional[str] = None):
        """
        Initialize the symbolic processor
        
        Parameters:
            grammar_path (str, optional): Path to the grammar JSON file
            patterns_path (str, optional): Path to the patterns JSON file
        """
        self.raga_grammar = RagaGrammar(grammar_path)
        self.tala_patterns = TalaPatterns(patterns_path)
    
    def analyze_composition(self, composition: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a composition
        
        Parameters:
            composition (dict): Composition to analyze
            
        Returns:
            dict: Analysis results
        """
        results = {
            "raga_analysis": None,
            "tala_analysis": None,
            "feedback": []
        }
        
        # Extract composition components
        raga_name = composition.get("raga")
        tala_name = composition.get("tala")
        swaras = composition.get("swaras", "")
        rhythm = composition.get("rhythm", "")
        
        # Analyze raga
        if raga_name and swaras:
            raga_rules = self.raga_grammar.get_raga_rules(raga_name)
            if raga_rules:
                # Validate the swaras against raga rules
                validation = self.raga_grammar.validate_phrase(swaras, raga_name)
                
                results["raga_analysis"] = {
                    "raga": raga_name,
                    "valid": validation["valid"],
                    "errors": validation["errors"],
                    "warnings": validation["warnings"]
                }
                
                # Add feedback
                for error in validation["errors"]:
                    results["feedback"].append(f"Raga Error: {error}")
                
                for warning in validation["warnings"]:
                    results["feedback"].append(f"Raga Suggestion: {warning}")
                
                # Add suggestions
                if "vadi" in raga_rules and "samvadi" in raga_rules:
                    results["feedback"].append(
                        f"Raga Tip: In {raga_name}, emphasize the vadi ({raga_rules['vadi']}) "
                        f"and samvadi ({raga_rules['samvadi']}) swaras for authentic expression."
                    )
                
                if "time" in raga_rules:
                    results["feedback"].append(
                        f"Raga Tip: {raga_name} is traditionally performed during {raga_rules['time']}."
                    )
        
        # Analyze tala
        if tala_name and rhythm:
            tala = self.tala_patterns.get_tala_pattern(tala_name)
            if tala:
                # Check if rhythm matches tala structure
                standard_bols = tala.get("bols", "")
                standard_sections = standard_bols.split("|")
                standard_sections = [section.strip() for section in standard_sections]
                
                rhythm_sections = rhythm.split("|")
                rhythm_sections = [section.strip() for section in rhythm_sections]
                
                tala_analysis = {
                    "tala": tala_name,
                    "valid": len(rhythm_sections) == len(standard_sections),
                    "errors": [],
                    "warnings": []
                }
                
                # Check section count
                if len(rhythm_sections) != len(standard_sections):
                    error = (
                        f"Rhythm has {len(rhythm_sections)} sections, "
                        f"but {tala_name} should have {len(standard_sections)} sections"
                    )
                    tala_analysis["errors"].append(error)
                    results["feedback"].append(f"Tala Error: {error}")
                
                # Check section lengths
                for i, (rhythm_section, standard_section) in enumerate(
                    zip(rhythm_sections, standard_sections)
                ):
                    rhythm_bols = rhythm_section.split()
                    standard_bols = standard_section.split()
                    
                    if len(rhythm_bols) != len(standard_bols):
                        warning = (
                            f"Section {i+1} has {len(rhythm_bols)} bols, "
                            f"but should have {len(standard_bols)} bols"
                        )
                        tala_analysis["warnings"].append(warning)
                        results["feedback"].append(f"Tala Suggestion: {warning}")
                
                results["tala_analysis"] = tala_analysis
                
                # Add suggestions
                results["feedback"].append(
                    f"Tala Tip: {tala_name} has a clap pattern of {tala.get('clap_pattern', '')}"
                )
        
        return results
    
    def generate_composition(self, raga_name: str, tala_name: str, length: int = 32) -> Dict[str, Any]:
        """
        Generate a composition
        
        Parameters:
            raga_name (str): Name of the raga
            tala_name (str): Name of the tala
            length (int): Length of the composition in swaras
            
        Returns:
            dict: Generated composition
        """
        # Get raga and tala
        raga_rules = self.raga_grammar.get_raga_rules(raga_name)
        tala = self.tala_patterns.get_tala_pattern(tala_name)
        
        if not raga_rules or not tala:
            return {}
        
        # Generate swaras
        swaras = self.raga_grammar.generate_phrase(raga_name, length)
        
        # Generate rhythm
        rhythm = self.tala_patterns.generate_rhythm_pattern(tala_name, variations=2)
        
        return {
            "raga": raga_name,
            "tala": tala_name,
            "swaras": swaras,
            "rhythm": rhythm
        }
    
    def convert_to_notation(self, composition: Dict[str, Any], notation_type: str = "sargam") -> str:
        """
        Convert a composition to notation
        
        Parameters:
            composition (dict): Composition to convert
            notation_type (str): Type of notation (sargam, western, etc.)
            
        Returns:
            str: Notation
        """
        swaras = composition.get("swaras", "")
        rhythm = composition.get("rhythm", "")
        
        if not swaras:
            return ""
        
        if notation_type == "sargam":
            # Sargam notation is already in the right format
            return swaras
        
        elif notation_type == "western":
            # Convert Sargam to Western notation
            sargam_to_western = {
                "S": "C",
                "r": "D♭",
                "R": "D",
                "g": "E♭",
                "G": "E",
                "m": "F",
                "M": "F♯",
                "P": "G",
                "d": "A♭",
                "D": "A",
                "n": "B♭",
                "N": "B",
                "S'": "C'"
            }
            
            western_notation = []
            for swara in swaras.split():
                base_swara = swara.rstrip("'").rstrip(".")
                octave_marker = ""
                
                if "'" in swara:
                    octave_marker = "'"
                elif "." in swara:
                    octave_marker = "."
                
                if base_swara in sargam_to_western:
                    western_notation.append(sargam_to_western[base_swara] + octave_marker)
                else:
                    western_notation.append(swara)
            
            return " ".join(western_notation)
        
        else:
            logger.warning(f"Unsupported notation type: {notation_type}")
            return swaras

def analyze_composition_symbolic(composition: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze a composition using symbolic processing
    
    Parameters:
        composition (dict): Composition to analyze
        
    Returns:
        dict: Analysis results
    """
    processor = SymbolicProcessor()
    return processor.analyze_composition(composition)

def generate_composition_symbolic(raga_name: str, tala_name: str, length: int = 32) -> Dict[str, Any]:
    """
    Generate a composition using symbolic processing
    
    Parameters:
        raga_name (str): Name of the raga
        tala_name (str): Name of the tala
        length (int): Length of the composition in swaras
        
    Returns:
        dict: Generated composition
    """
    processor = SymbolicProcessor()
    return processor.generate_composition(raga_name, tala_name, length)

def convert_to_notation(composition: Dict[str, Any], notation_type: str = "sargam") -> str:
    """
    Convert a composition to notation
    
    Parameters:
        composition (dict): Composition to convert
        notation_type (str): Type of notation (sargam, western, etc.)
        
    Returns:
        str: Notation
    """
    processor = SymbolicProcessor()
    return processor.convert_to_notation(composition, notation_type)