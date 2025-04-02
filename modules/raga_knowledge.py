"""
Raga Knowledge Base for RagaVani application

This module provides information about Indian classical ragas,
including their structure, characteristics, and classifications.
"""

# Database of ragas with their properties
RAGA_DATABASE = {
    "Yaman": {
        "thaat": "Kalyan",
        "notes": {
            "aroha": ["S", "R", "G", "M#", "P", "D", "N", "S'"],
            "avaroha": ["S'", "N", "D", "P", "M#", "G", "R", "S"],
        },
        "vadi": "G",
        "samvadi": "N",
        "pakad": "N R G M# D N R' S'",
        "description": "Raga Yaman is one of the fundamental ragas in Hindustani classical music. It has a serene and peaceful character, often evoking feelings of devotion and tranquility. It is traditionally performed in the early evening hours and is known for its distinctive ascending pattern that skips the natural Ma.",
        "time": "Early Evening (7-10 PM)",
        "mood": ["Serene", "Peaceful", "Devotional", "Romantic"],
        "similar_ragas": ["Yaman Kalyan", "Shuddha Kalyan", "Bhupali"],
        "famous_compositions": [
            "Eri Ali Piya Bina by Ustad Amir Khan",
            "Ab To Jag Jani by Bhimsen Joshi",
            "Many instrumental gats in sitar and sarod traditions"
        ]
    },
    "Bhairav": {
        "thaat": "Bhairav",
        "notes": {
            "aroha": ["S", "r", "G", "M", "P", "d", "N", "S'"],
            "avaroha": ["S'", "N", "d", "P", "M", "G", "r", "S"],
        },
        "vadi": "D",
        "samvadi": "R",
        "pakad": "S r S d N S r G M d N S",
        "description": "Raga Bhairav is one of the oldest ragas in Hindustani classical music, traditionally performed in the early morning hours. It is associated with Lord Shiva and has a profound, serious, and meditative quality. The flat second (komal Re) and flat sixth (komal Dha) give it its distinctive character.",
        "time": "Early Morning (6-9 AM)",
        "mood": ["Serious", "Profound", "Devotional", "Meditative"],
        "similar_ragas": ["Bhairav Bahar", "Ahir Bhairav", "Nat Bhairav"],
        "famous_compositions": [
            "Jago Mohan Pyare by Bhimsen Joshi",
            "Bhavani Dayani by Kishori Amonkar",
            "Various dhrupad compositions by Dagar Brothers"
        ]
    },
    "Bhairavi": {
        "thaat": "Bhairavi",
        "notes": {
            "aroha": ["S", "r", "g", "M", "P", "d", "n", "S'"],
            "avaroha": ["S'", "n", "d", "P", "M", "g", "r", "S"],
        },
        "vadi": "M",
        "samvadi": "S",
        "pakad": "S g M P d M g r S",
        "description": "Raga Bhairavi is often called the 'queen of ragas' and is traditionally performed as the concluding raga in a concert. It has all flat notes (komal) except for Sa and Pa, giving it a deeply emotional and sometimes melancholic quality. It is versatile and can express various moods from devotion to pathos.",
        "time": "Morning (especially concluding performances)",
        "mood": ["Devotional", "Melancholic", "Profound", "Serene"],
        "similar_ragas": ["Bilaskhani Todi", "Kafi", "Pilu"],
        "famous_compositions": [
            "Babul Mora by K.L. Saigal",
            "Jamuna Ke Teer by Bhimsen Joshi",
            "Various thumri compositions"
        ]
    },
    "Darbari Kanada": {
        "thaat": "Asavari",
        "notes": {
            "aroha": ["S", "R", "g", "M", "P", "d", "n", "S'"],
            "avaroha": ["S'", "n", "d", "P", "M", "g", "M", "R", "S"],
        },
        "vadi": "R",
        "samvadi": "P",
        "pakad": "R g R S n d P M g M R S",
        "description": "Raga Darbari Kanada is a profound and majestic raga, believed to have been introduced to the court of Emperor Akbar by Tansen. It is characterized by its distinctive slow and serene development, with signature gamaks (oscillations) on komal gandhar (g) and komal nishad (n). It creates a deep, serious atmosphere of contemplation.",
        "time": "Late Night (10 PM - 1 AM)",
        "mood": ["Majestic", "Profound", "Serious", "Meditative"],
        "similar_ragas": ["Adana", "Jaunpuri", "Shahana Kanada"],
        "famous_compositions": [
            "Ankhiyan Haari by Kishori Amonkar",
            "Multiple khayal compositions by Ustad Amir Khan",
            "Various instrumental compositions by Ustad Ali Akbar Khan"
        ]
    },
    "Malkauns": {
        "thaat": "Bhairavi",
        "notes": {
            "aroha": ["S", "g", "M", "d", "n", "S'"],
            "avaroha": ["S'", "n", "d", "M", "g", "S"],
        },
        "vadi": "M",
        "samvadi": "S",
        "pakad": "n d M g M g S",
        "description": "Raga Malkauns is one of the oldest ragas in Indian classical music, believed to have the power to charm serpents and light oil lamps with its resonance. It is a pentatonic raga with all flat notes except Sa, creating a mysterious and profound atmosphere. It is traditionally performed in the late night hours.",
        "time": "Late Night (midnight)",
        "mood": ["Profound", "Serious", "Mystical", "Intense"],
        "similar_ragas": ["Chandrakauns", "Bageshri", "Nandkauns"],
        "famous_compositions": [
            "Man Mohana Bade Jhoothe by Kishori Amonkar",
            "Multiple instrumental compositions by Nikhil Banerjee",
            "Various khayal compositions by Kumar Gandharva"
        ]
    },
    "Todi": {
        "thaat": "Todi",
        "notes": {
            "aroha": ["S", "r", "g", "M#", "P", "d", "N", "S'"],
            "avaroha": ["S'", "N", "d", "P", "M#", "g", "r", "S"],
        },
        "vadi": "d",
        "samvadi": "r",
        "pakad": "r g M# g r S",
        "description": "Raga Todi is considered one of the most complex and profound ragas in Hindustani classical music. It has a contemplative and serious mood, with a distinctive combination of flat second (komal Re), flat third (komal Ga), sharp fourth (tivra Ma) and flat sixth (komal Dha). It requires immense skill to perform correctly.",
        "time": "Late Morning (10 AM - 1 PM)",
        "mood": ["Serious", "Profound", "Complex", "Meditative"],
        "similar_ragas": ["Multani", "Gurjari Todi", "Miyan ki Todi"],
        "famous_compositions": [
            "Langar Kankariya by Ustad Faiyaz Khan",
            "Ektal compositions by Ustad Amir Khan",
            "Various instrumental compositions by Pandit Ravi Shankar"
        ]
    },
    "Bihag": {
        "thaat": "Bilawal",
        "notes": {
            "aroha": ["S", "G", "M", "P", "N", "S'"],
            "avaroha": ["S'", "N", "P", "M", "G", "R", "S"],
        },
        "vadi": "G",
        "samvadi": "N",
        "pakad": "G M P G M R S N S",
        "description": "Raga Bihag is a melodious and romantic raga performed in the first part of the night. It has a sweet and playful character, often evoking feelings of love and joy. It is characterized by its distinctive vakra (zigzag) movements and the prominence of Gandhar (G) and Nishad (N). It's popular for both classical and semi-classical forms.",
        "time": "First part of night (9 PM - midnight)",
        "mood": ["Romantic", "Joyful", "Sweet", "Playful"],
        "similar_ragas": ["Bihagda", "Champak", "Nat Bihag"],
        "famous_compositions": [
            "Kaise Sukh Sove by Rashid Khan",
            "Jamuna Kinare Mora Gaon by Bhimsen Joshi",
            "Various thumri compositions by Girija Devi"
        ]
    },
    "Bageshri": {
        "thaat": "Kafi",
        "notes": {
            "aroha": ["S", "R", "g", "M", "P", "D", "n", "S'"],
            "avaroha": ["S'", "n", "D", "P", "M", "g", "R", "S"],
        },
        "vadi": "M",
        "samvadi": "S",
        "pakad": "S g M P, M g R S",
        "description": "Raga Bageshri is a beautiful and lyrical raga performed in the middle to late night hours. It evokes feelings of yearning and longing, often associated with the emotion of separation from a loved one (viraha). The komal gandhar (g) and komal nishad (n) give it its distinctive character and emotional appeal.",
        "time": "Late night (midnight - 3 AM)",
        "mood": ["Yearning", "Romantic", "Sweet", "Melancholic"],
        "similar_ragas": ["Bhimpalasi", "Chhayanat", "Malkauns"],
        "famous_compositions": [
            "Kahe Sataye Mohey by Bhimsen Joshi",
            "Piya Bina by Kishori Amonkar",
            "Several thumri and ghazal compositions"
        ]
    },
    "Miyan Ki Malhar": {
        "thaat": "Kafi",
        "notes": {
            "aroha": ["S", "R", "g", "M", "P", "D", "N", "S'"],
            "avaroha": ["S'", "N", "D", "P", "M", "g", "R", "S"],
        },
        "vadi": "M",
        "samvadi": "S",
        "pakad": "R g R S N. D. N. S, R M P N D P M g R S",
        "description": "Raga Miyan Ki Malhar is a monsoon raga, traditionally associated with the rainy season. It has a joyful yet complex character, evoking the mood of clouds gathering and the relief of rain after summer heat. It was possibly created by Tansen and features distinctive phrases that capture the sound of thunder and raindrops.",
        "time": "Evening/Night during monsoon season",
        "mood": ["Joyful", "Majestic", "Romantic", "Serene"],
        "similar_ragas": ["Gaud Malhar", "Ramdasi Malhar", "Megh"],
        "famous_compositions": [
            "Barsan Lagi by Kishori Amonkar",
            "Multiple khayal compositions by Pandit Jasraj",
            "Various instrumental compositions by Ustad Vilayat Khan"
        ]
    },
    "Kedar": {
        "thaat": "Kalyan",
        "notes": {
            "aroha": ["S", "M", "P", "D", "P", "M", "P", "S'"],
            "avaroha": ["S'", "N", "D", "P", "M", "G", "R", "S"],
        },
        "vadi": "M",
        "samvadi": "S",
        "pakad": "M P D P, M P, M G, R S",
        "description": "Raga Kedar is a serene and devotional raga, often associated with Lord Shiva. It has a distinctive ascending pattern that skips several notes and features vakra (zigzag) movements. The madhyam (M) is prominent and creates the peaceful and devotional mood that characterizes this raga.",
        "time": "Late evening (9 PM - midnight)",
        "mood": ["Devotional", "Serene", "Peaceful", "Profound"],
        "similar_ragas": ["Chandrakauns", "Hameer", "Kamod"],
        "famous_compositions": [
            "Jamuna Ke Teer by Pt. Bhimsen Joshi",
            "Various dhrupad compositions by Gundecha Brothers",
            "Instrumental compositions by Ustad Amjad Ali Khan"
        ]
    }
}

def get_raga_info(raga_name):
    """
    Get information about a raga
    
    Parameters:
        raga_name (str): Name of the raga
    
    Returns:
        dict or None: Dictionary containing raga information, or None if not found
    """
    # Try exact match first
    if raga_name in RAGA_DATABASE:
        return RAGA_DATABASE[raga_name]
    
    # Try case-insensitive match
    for name, info in RAGA_DATABASE.items():
        if name.lower() == raga_name.lower():
            return info
    
    return None

def get_all_ragas():
    """
    Get a list of all available ragas
    
    Returns:
        list: List of raga names
    """
    return list(RAGA_DATABASE.keys())

def get_raga_by_mood(mood):
    """
    Get ragas matching a specific mood
    
    Parameters:
        mood (str): Mood to search for
    
    Returns:
        list: List of raga names matching the mood
    """
    matching_ragas = []
    mood = mood.lower()
    
    for raga_name, raga_info in RAGA_DATABASE.items():
        if "mood" in raga_info:
            for raga_mood in raga_info["mood"]:
                if mood in raga_mood.lower():
                    matching_ragas.append(raga_name)
                    break
    
    return matching_ragas

def get_raga_by_time(time_of_day):
    """
    Get ragas traditionally performed at a specific time of day
    
    Parameters:
        time_of_day (str): Time of day to search for
    
    Returns:
        list: List of raga names matching the time of day
    """
    matching_ragas = []
    time_of_day = time_of_day.lower()
    
    for raga_name, raga_info in RAGA_DATABASE.items():
        if "time" in raga_info and time_of_day in raga_info["time"].lower():
            matching_ragas.append(raga_name)
    
    return matching_ragas

def get_raga_by_notes(notes_present, notes_absent=None):
    """
    Get ragas that contain specific notes and exclude others
    
    Parameters:
        notes_present (list): List of notes that must be present in the raga
        notes_absent (list, optional): List of notes that must be absent from the raga
    
    Returns:
        list: List of raga names matching the criteria
    """
    matching_ragas = []
    
    if notes_absent is None:
        notes_absent = []
    
    for raga_name, raga_info in RAGA_DATABASE.items():
        if "notes" not in raga_info:
            continue
        
        # Get all notes in the raga (both aroha and avaroha)
        all_notes = set(raga_info["notes"]["aroha"] + raga_info["notes"]["avaroha"])
        
        # Check if all required notes are present
        if all(note in all_notes for note in notes_present):
            # Check if all excluded notes are absent
            if all(note not in all_notes for note in notes_absent):
                matching_ragas.append(raga_name)
    
    return matching_ragas

def compare_ragas(raga1, raga2):
    """
    Compare two ragas and highlight similarities and differences
    
    Parameters:
        raga1 (str): Name of first raga
        raga2 (str): Name of second raga
    
    Returns:
        dict: Dictionary containing comparison results
    """
    info1 = get_raga_info(raga1)
    info2 = get_raga_info(raga2)
    
    if not info1 or not info2:
        return None
    
    comparison = {
        "raga1": raga1,
        "raga2": raga2,
        "similarities": {},
        "differences": {}
    }
    
    # Compare thaats
    if info1.get("thaat") == info2.get("thaat"):
        comparison["similarities"]["thaat"] = info1.get("thaat")
    else:
        comparison["differences"]["thaat"] = {
            "raga1": info1.get("thaat"),
            "raga2": info2.get("thaat")
        }
    
    # Compare notes
    if "notes" in info1 and "notes" in info2:
        notes1 = set(info1["notes"]["aroha"] + info1["notes"]["avaroha"])
        notes2 = set(info2["notes"]["aroha"] + info2["notes"]["avaroha"])
        
        comparison["similarities"]["common_notes"] = list(notes1.intersection(notes2))
        comparison["differences"]["unique_notes"] = {
            "raga1": list(notes1 - notes2),
            "raga2": list(notes2 - notes1)
        }
    
    # Compare vadi and samvadi
    if info1.get("vadi") == info2.get("vadi"):
        comparison["similarities"]["vadi"] = info1.get("vadi")
    else:
        comparison["differences"]["vadi"] = {
            "raga1": info1.get("vadi"),
            "raga2": info2.get("vadi")
        }
    
    if info1.get("samvadi") == info2.get("samvadi"):
        comparison["similarities"]["samvadi"] = info1.get("samvadi")
    else:
        comparison["differences"]["samvadi"] = {
            "raga1": info1.get("samvadi"),
            "raga2": info2.get("samvadi")
        }
    
    # Compare time of day
    if info1.get("time") == info2.get("time"):
        comparison["similarities"]["time"] = info1.get("time")
    else:
        comparison["differences"]["time"] = {
            "raga1": info1.get("time"),
            "raga2": info2.get("time")
        }
    
    # Compare mood
    if "mood" in info1 and "mood" in info2:
        mood1 = set(info1["mood"])
        mood2 = set(info2["mood"])
        
        comparison["similarities"]["common_moods"] = list(mood1.intersection(mood2))
        comparison["differences"]["unique_moods"] = {
            "raga1": list(mood1 - mood2),
            "raga2": list(mood2 - mood1)
        }
    
    return comparison