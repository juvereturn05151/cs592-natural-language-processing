"""
File Name:    play_configs.py
Author(s):    Ju-ve Chankasemporn
Copyright:    (c) 2025 DigiPen Institute of Technology. All rights reserved.
"""

PLAY_CONFIGS = {

    "Shakespeare_Macbeth.txt": {
        "known_gpes": {
            "Scotland", "England", "Forres", "Inverness", "Fife",
            "Dunsinane", "Northumberland", "Norway", "Aleppo",
            "Saint Colme's Inch", "Birnam Wood", "Birnam"
        },
        "known_locations": {
            "the Castle", "the Palace", "the heath", "A dark Cave",
            "the Plain", "the field", "a Wood", "the Court", "A Camp"
        },
        "known_titles": {
            "Thane of Glamis", "Thane of Cawdor", "Thane of Fife",
            "King of Scotland", "Earl of Northumberland",
            "General of the English Forces"
        },
        "male_characters": {
            "MACBETH", "DUNCAN", "MALCOLM", "DONALBAIN", "BANQUO",
            "MACDUFF", "LENNOX", "ROSS", "MENTEITH", "ANGUS",
            "CAITHNESS", "FLEANCE", "SIWARD", "YOUNG SIWARD", "SEYTON", "BOY"
        },
        "female_characters": {
            "LADY MACBETH", "LADY MACDUFF", "HECATE"
        },
        "relationships": [
            ("MACBETH",      "PERSON", "MARRIED_TO",     "LADY MACBETH",  "PERSON"),
            ("MACDUFF",      "PERSON", "MARRIED_TO",     "LADY MACDUFF",  "PERSON"),
            ("FLEANCE",      "PERSON", "SON_OF",         "BANQUO",        "PERSON"),
            ("MALCOLM",      "PERSON", "SON_OF",         "DUNCAN",        "PERSON"),
            ("DONALBAIN",    "PERSON", "SON_OF",         "DUNCAN",        "PERSON"),
            ("BOY",          "PERSON", "SON_OF",         "MACDUFF",       "PERSON"),
            ("YOUNG SIWARD", "PERSON", "SON_OF",         "SIWARD",        "PERSON"),
            ("MACBETH",      "PERSON", "KILLS",          "DUNCAN",        "PERSON"),
            ("MACBETH",      "PERSON", "KILLS",          "BANQUO",        "PERSON"),
            ("MACBETH",      "PERSON", "KILLS",          "YOUNG SIWARD",  "PERSON"),
            ("MACDUFF",      "PERSON", "KILLS",          "MACBETH",       "PERSON"),
            ("MACBETH",      "PERSON", "KILLS_OFFSTAGE", "LADY MACDUFF",  "PERSON"),
            ("MACBETH",      "PERSON", "KILLS_OFFSTAGE", "BOY",           "PERSON"),
            ("LADY MACBETH", "PERSON", "KILLS_OFFSTAGE", "DUNCAN",        "PERSON"),
            ("MACBETH",      "PERSON", "LOYAL_TO",       "DUNCAN",        "PERSON"),
            ("MACBETH",      "PERSON", "BETRAYS",        "DUNCAN",        "PERSON"),
            ("BANQUO",       "PERSON", "LOYAL_TO",       "DUNCAN",        "PERSON"),
            ("MACDUFF",      "PERSON", "LOYAL_TO",       "MALCOLM",       "PERSON"),
            ("ROSS",         "PERSON", "LOYAL_TO",       "DUNCAN",        "PERSON"),
            ("DUNCAN",       "PERSON", "RULES",          "Scotland",      "GPE"),
            ("MACBETH",      "PERSON", "RULES",          "Scotland",      "GPE"),
            ("MALCOLM",      "PERSON", "RULES",          "Scotland",      "GPE"),
            ("SIWARD",       "PERSON", "RULES",          "Northumberland","GPE"),
            ("MACBETH",      "PERSON", "HOLDS_TITLE",    "Thane of Glamis",   "TITLE"),
            ("MACBETH",      "PERSON", "HOLDS_TITLE",    "Thane of Cawdor",   "TITLE"),
            ("MACBETH",      "PERSON", "HOLDS_TITLE",    "King of Scotland",  "TITLE"),
            ("DUNCAN",       "PERSON", "HOLDS_TITLE",    "King of Scotland",  "TITLE"),
            ("MACDUFF",      "PERSON", "HOLDS_TITLE",    "Thane of Fife",     "TITLE"),
            ("SIWARD",       "PERSON", "HOLDS_TITLE",    "Earl of Northumberland", "TITLE"),
            ("Inverness",    "GPE",    "LOCATED_IN",     "Scotland",      "GPE"),
            ("Forres",       "GPE",    "LOCATED_IN",     "Scotland",      "GPE"),
            ("Fife",         "GPE",    "LOCATED_IN",     "Scotland",      "GPE"),
            ("Dunsinane",    "GPE",    "LOCATED_IN",     "Scotland",      "GPE"),
            ("Birnam Wood",  "GPE",    "LOCATED_IN",     "Scotland",      "GPE"),
        ]
    },

    "Shakespeare_Romeo_and_Juliet.txt": {
        "known_gpes": {
            "Verona", "Mantua", "Italy", "Rome"
        },
        "known_locations": {
            "Capulet's Garden", "Friar Lawrence's Cell", "Juliet's Chamber",
            "A public place", "A churchyard", "A Street"
        },
        "known_titles": {
            "Prince of Verona", "head of a Veronese family",
            "Friar", "Nurse"
        },
        "male_characters": {
            "ROMEO", "MONTAGUE", "BENVOLIO", "MERCUTIO", "TYBALT",
            "CAPULET", "PARIS", "FRIAR LAWRENCE", "FRIAR JOHN",
            "BALTHASAR", "SAMPSON", "GREGORY", "ABRAM", "PETER",
            "ESCALUS", "CHORUS"
        },
        "female_characters": {
            "JULIET", "LADY CAPULET", "LADY MONTAGUE", "NURSE"
        },
        "relationships": [
            ("ROMEO",          "PERSON", "MARRIED_TO",     "JULIET",         "PERSON"),
            ("ROMEO",          "PERSON", "SON_OF",         "MONTAGUE",       "PERSON"),
            ("JULIET",         "PERSON", "DAUGHTER_OF",    "CAPULET",        "PERSON"),
            ("BENVOLIO",       "PERSON", "NEPHEW_OF",      "MONTAGUE",       "PERSON"),
            ("TYBALT",         "PERSON", "NEPHEW_OF",      "LADY CAPULET",   "PERSON"),
            ("ROMEO",          "PERSON", "KILLS",          "TYBALT",         "PERSON"),
            ("ROMEO",          "PERSON", "KILLS",          "PARIS",          "PERSON"),
            ("TYBALT",         "PERSON", "KILLS",          "MERCUTIO",       "PERSON"),
            ("ROMEO",          "PERSON", "LOVES",          "JULIET",         "PERSON"),
            ("JULIET",         "PERSON", "LOVES",          "ROMEO",          "PERSON"),
            ("PARIS",          "PERSON", "LOVES",          "JULIET",         "PERSON"),
            ("MONTAGUE",       "PERSON", "FEUD_WITH",      "CAPULET",        "PERSON"),
            ("CAPULET",        "PERSON", "FEUD_WITH",      "MONTAGUE",       "PERSON"),
            ("FRIAR LAWRENCE", "PERSON", "ALLIES_WITH",    "ROMEO",          "PERSON"),
            ("ESCALUS",        "PERSON", "RULES",          "Verona",         "GPE"),
            ("MONTAGUE",       "PERSON", "BASED_IN",       "Verona",         "GPE"),
            ("CAPULET",        "PERSON", "BASED_IN",       "Verona",         "GPE"),
            ("ROMEO",          "PERSON", "HOLDS_TITLE",    "Prince of Verona","TITLE"),
            ("ESCALUS",        "PERSON", "HOLDS_TITLE",    "Prince of Verona","TITLE"),
            ("Mantua",         "GPE",    "LOCATED_IN",     "Italy",          "GPE"),
            ("Verona",         "GPE",    "LOCATED_IN",     "Italy",          "GPE"),
        ]
    },

    "Shakespeare_Midsummer_Nights_Dream.txt": {
        "known_gpes": {
            "Athens", "Greece"
        },
        "known_locations": {
            "A wood near Athens", "the wood", "the Palace of Theseus",
            "A Room in a Cottage", "Quince's House"
        },
        "known_titles": {
            "Duke of Athens", "Queen of the Amazons",
            "King of the Fairies", "Queen of the Fairies",
            "Master of the Revels"
        },
        "male_characters": {
            "THESEUS", "EGEUS", "LYSANDER", "DEMETRIUS",
            "PHILOSTRATE", "QUINCE", "SNUG", "BOTTOM",
            "FLUTE", "SNOUT", "STARVELING", "OBERON", "PUCK"
        },
        "female_characters": {
            "HIPPOLYTA", "HERMIA", "HELENA", "TITANIA",
            "PEASEBLOSSOM", "COBWEB", "MOTH", "MUSTARDSEED"
        },
        "relationships": [
            ("THESEUS",    "PERSON", "MARRIED_TO",   "HIPPOLYTA",  "PERSON"),
            ("OBERON",     "PERSON", "MARRIED_TO",   "TITANIA",    "PERSON"),
            ("HERMIA",     "PERSON", "DAUGHTER_OF",  "EGEUS",      "PERSON"),
            ("LYSANDER",   "PERSON", "LOVES",        "HERMIA",     "PERSON"),
            ("DEMETRIUS",  "PERSON", "LOVES",        "HERMIA",     "PERSON"),
            ("DEMETRIUS",  "PERSON", "LOVES",        "HELENA",     "PERSON"),
            ("HELENA",     "PERSON", "LOVES",        "DEMETRIUS",  "PERSON"),
            ("HERMIA",     "PERSON", "LOVES",        "LYSANDER",   "PERSON"),
            ("TITANIA",    "PERSON", "LOVES",        "BOTTOM",     "PERSON"),
            ("PUCK",       "PERSON", "SERVES",       "OBERON",     "PERSON"),
            ("THESEUS",    "PERSON", "RULES",        "Athens",     "GPE"),
            ("OBERON",     "PERSON", "RULES",        "Fairy Kingdom", "LOCATION"),
            ("THESEUS",    "PERSON", "HOLDS_TITLE",  "Duke of Athens",        "TITLE"),
            ("HIPPOLYTA",  "PERSON", "HOLDS_TITLE",  "Queen of the Amazons",  "TITLE"),
            ("OBERON",     "PERSON", "HOLDS_TITLE",  "King of the Fairies",   "TITLE"),
            ("TITANIA",    "PERSON", "HOLDS_TITLE",  "Queen of the Fairies",  "TITLE"),
        ]
    },

    "Shakespeare_Much_Ado_About_Nothing.txt": {
        "known_gpes": {
            "Messina", "Arragon", "Florence", "Padua", "Italy"
        },
        "known_locations": {
            "Leonato's House", "Leonato's Garden", "A Street",
            "The Inside of a Church", "A Prison"
        },
        "known_titles": {
            "Prince of Arragon", "Governor of Messina",
            "Lord of Florence", "Lord of Padua", "Constable"
        },
        "male_characters": {
            "DON PEDRO", "DON JOHN", "CLAUDIO", "BENEDICK",
            "LEONATO", "ANTONIO", "BALTHASAR", "BORACHIO",
            "CONRADE", "DOGBERRY", "VERGES", "FRIAR FRANCIS"
        },
        "female_characters": {
            "HERO", "BEATRICE", "MARGARET", "URSULA"
        },
        "relationships": [
            ("BENEDICK",   "PERSON", "MARRIED_TO",   "BEATRICE",   "PERSON"),
            ("CLAUDIO",    "PERSON", "MARRIED_TO",   "HERO",       "PERSON"),
            ("HERO",       "PERSON", "DAUGHTER_OF",  "LEONATO",    "PERSON"),
            ("BEATRICE",   "PERSON", "NIECE_OF",     "LEONATO",    "PERSON"),
            ("ANTONIO",    "PERSON", "BROTHER_OF",   "LEONATO",    "PERSON"),
            ("DON JOHN",   "PERSON", "BROTHER_OF",   "DON PEDRO",  "PERSON"),
            ("BENEDICK",   "PERSON", "LOVES",        "BEATRICE",   "PERSON"),
            ("BEATRICE",   "PERSON", "LOVES",        "BENEDICK",   "PERSON"),
            ("CLAUDIO",    "PERSON", "LOVES",        "HERO",       "PERSON"),
            ("DON JOHN",   "PERSON", "DECEIVES",     "CLAUDIO",    "PERSON"),
            ("BORACHIO",   "PERSON", "ALLIES_WITH",  "DON JOHN",   "PERSON"),
            ("CONRADE",    "PERSON", "ALLIES_WITH",  "DON JOHN",   "PERSON"),
            ("DON PEDRO",  "PERSON", "RULES",        "Arragon",    "GPE"),
            ("LEONATO",    "PERSON", "RULES",        "Messina",    "GPE"),
            ("DON PEDRO",  "PERSON", "HOLDS_TITLE",  "Prince of Arragon",  "TITLE"),
            ("LEONATO",    "PERSON", "HOLDS_TITLE",  "Governor of Messina","TITLE"),
            ("DOGBERRY",   "PERSON", "HOLDS_TITLE",  "Constable",          "TITLE"),
            ("Messina",    "GPE",    "LOCATED_IN",   "Italy",      "GPE"),
            ("Florence",   "GPE",    "LOCATED_IN",   "Italy",      "GPE"),
            ("Padua",      "GPE",    "LOCATED_IN",   "Italy",      "GPE"),
            ("Arragon",    "GPE",    "LOCATED_IN",   "Italy",      "GPE"),
        ]
    },
}
