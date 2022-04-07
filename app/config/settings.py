NUM_TOP_K = 5
PATCH_SIZE = 105
NUM_PATCHES = 5
MODEL_PATH = "models/font117_vgg16.pt"

FONT_LABEL_TO_META = {
    0: {
        "fontName": "kaiseiopti",
        "fontNameJa": "解星オプティ",
        "fontNameEn": "Kaisei Opti",
        "fontWeight": 400,
    },
    1: {
        "fontName": "kaiseiopti",
        "fontNameJa": "解星オプティ",
        "fontNameEn": "Kaisei Opti",
        "fontWeight": 700,
    },
    2: {
        "fontName": "kaiseiopti",
        "fontNameJa": "解星オプティ",
        "fontNameEn": "Kaisei Opti",
        "fontWeight": 500,
    },
    3: {
        "fontName": "kaiseidecol",
        "fontNameJa": "解星デコール",
        "fontNameEn": "Kaisei Decol",
        "fontWeight": 400,
    },
    4: {
        "fontName": "kaiseidecol",
        "fontNameJa": "解星デコール",
        "fontNameEn": "Kaisei Decol",
        "fontWeight": 500,
    },
    5: {
        "fontName": "kaiseidecol",
        "fontNameJa": "解星デコール",
        "fontNameEn": "Kaisei Decol",
        "fontWeight": 700,
    },
    6: {
        "fontName": "kaiseiharunoumi",
        "fontNameJa": "解星 春の海",
        "fontNameEn": "Kaisei HarunoUmi",
        "fontWeight": 400,
    },
    7: {
        "fontName": "kaiseiharunoumi",
        "fontNameJa": "解星 春の海",
        "fontNameEn": "Kaisei HarunoUmi",
        "fontWeight": 500,
    },
    8: {
        "fontName": "kaiseiharunoumi",
        "fontNameJa": "解星 春の海",
        "fontNameEn": "Kaisei HarunoUmi",
        "fontWeight": 700,
    },
    9: {
        "fontName": "kaiseitokumin",
        "fontNameJa": "解星 特ミン",
        "fontNameEn": "Kaisei Tokumin",
        "fontWeight": 800,
    },
    10: {
        "fontName": "kaiseitokumin",
        "fontNameJa": "解星 特ミン",
        "fontNameEn": "Kaisei Tokumin",
        "fontWeight": 700,
    },
    11: {
        "fontName": "kaiseitokumin",
        "fontNameJa": "解星 特ミン",
        "fontNameEn": "Kaisei Tokumin",
        "fontWeight": 500,
    },
    12: {
        "fontName": "kaiseitokumin",
        "fontNameJa": "解星 特ミン",
        "fontNameEn": "Kaisei Tokumin",
        "fontWeight": 400,
    },
    13: {
        "fontName": "hinamincho",
        "fontNameJa": "ひな明朝",
        "fontNameEn": "Hina Mincho",
        "fontWeight": 400,
    },
    14: {
        "fontName": "kleeone",
        "fontNameJa": "クレー One",
        "fontNameEn": "Klee One",
        "fontWeight": 400,
    },
    15: {
        "fontName": "kleeone",
        "fontNameJa": "クレー One",
        "fontNameEn": "Klee One",
        "fontWeight": 600,
    },
    16: {
        "fontName": "rampartone",
        "fontNameJa": "ランパート One",
        "fontNameEn": "Rampart One",
        "fontWeight": 400,
    },
    17: {
        "fontName": "shipporimincho",
        "fontNameJa": "しっぽり明朝",
        "fontNameEn": "Shippori Mincho",
        "fontWeight": 700,
    },
    18: {
        "fontName": "shipporimincho",
        "fontNameJa": "しっぽり明朝",
        "fontNameEn": "Shippori Mincho",
        "fontWeight": 600,
    },
    19: {
        "fontName": "shipporimincho",
        "fontNameJa": "しっぽり明朝",
        "fontNameEn": "Shippori Mincho",
        "fontWeight": 400,
    },
    20: {
        "fontName": "shipporimincho",
        "fontNameJa": "しっぽり明朝",
        "fontNameEn": "Shippori Mincho",
        "fontWeight": 500,
    },
    21: {
        "fontName": "shipporimincho",
        "fontNameJa": "しっぽり明朝",
        "fontNameEn": "Shippori Mincho",
        "fontWeight": 800,
    },
    22: {
        "fontName": "sawarabigothic",
        "fontNameJa": "さわらびゴシック",
        "fontNameEn": "Sawarabi Gothic",
        "fontWeight": 400,
    },
    23: {
        "fontName": "sawarabimincho",
        "fontNameJa": "さわらび明朝",
        "fontNameEn": "Sawarabi Mincho",
        "fontWeight": 400,
    },
    24: {
        "fontName": "newtegomin",
        "fontNameJa": "ニューテゴミン",
        "fontNameEn": "New Tegomin",
        "fontWeight": 400,
    },
    25: {
        "fontName": "kiwimaru",
        "fontNameJa": "キウイ丸",
        "fontNameEn": "Kiwi Maru",
        "fontWeight": 500,
    },
    26: {
        "fontName": "kiwimaru",
        "fontNameJa": "キウイ丸",
        "fontNameEn": "Kiwi Maru",
        "fontWeight": 300,
    },
    27: {
        "fontName": "kiwimaru",
        "fontNameJa": "キウイ丸",
        "fontNameEn": "Kiwi Maru",
        "fontWeight": 400,
    },
    28: {
        "fontName": "delagothicone",
        "fontNameJa": "デラゴシック",
        "fontNameEn": "Dela Gothic One",
        "fontWeight": 400,
    },
    29: {
        "fontName": "yomogi",
        "fontNameJa": "Yomogi",
        "fontNameEn": "Yomogi",
        "fontWeight": 400,
    },
    30: {
        "fontName": "hachimarupop",
        "fontNameJa": "はちまるポップ",
        "fontNameEn": "Hachi Maru Pop",
        "fontWeight": 400,
    },
    31: {
        "fontName": "pottaone",
        "fontNameJa": "ポッタ",
        "fontNameEn": "Potta One",
        "fontWeight": 400,
    },
    32: {
        "fontName": "stick",
        "fontNameJa": "ステッキ",
        "fontNameEn": "Stick",
        "fontWeight": 400,
    },
    33: {
        "fontName": "rocknrollone",
        "fontNameJa": "ロックンロール One",
        "fontNameEn": "RocknRoll One",
        "fontWeight": 400,
    },
    34: {
        "fontName": "reggaeone",
        "fontNameJa": "レゲエ One",
        "fontNameEn": "Reggae One",
        "fontWeight": 400,
    },
    35: {
        "fontName": "trainone",
        "fontNameJa": "トレイン One",
        "fontNameEn": "Train One",
        "fontWeight": 400,
    },
    36: {
        "fontName": "dotgothic16",
        "fontNameJa": "ドットゴシック16",
        "fontNameEn": "DotGothic16",
        "fontWeight": 400,
    },
    37: {
        "fontName": "yuseimagic",
        "fontNameJa": "YuseiMagic",
        "fontNameEn": "Yusei Magic",
        "fontWeight": 400,
    },
    38: {
        "fontName": "kosugi",
        "fontNameJa": "小杉フォント",
        "fontNameEn": "Kosugi",
        "fontWeight": 400,
    },
    39: {
        "fontName": "kosugimaru",
        "fontNameJa": "小杉丸フォント",
        "fontNameEn": "Kosugi Maru",
        "fontWeight": 400,
    },
    40: {
        "fontName": "mplus1",
        "fontNameJa": "Mplus 1",
        "fontNameEn": "M PLUS 1",
        "fontWeight": 500,
    },
    41: {
        "fontName": "mplus1",
        "fontNameJa": "Mplus 1",
        "fontNameEn": "M PLUS 1",
        "fontWeight": 400,
    },
    42: {
        "fontName": "mplus1",
        "fontNameJa": "Mplus 1",
        "fontNameEn": "M PLUS 1",
        "fontWeight": 900,
    },
    43: {
        "fontName": "mplus1",
        "fontNameJa": "Mplus 1",
        "fontNameEn": "M PLUS 1",
        "fontWeight": 800,
    },
    44: {
        "fontName": "mplus1",
        "fontNameJa": "Mplus 1",
        "fontNameEn": "M PLUS 1",
        "fontWeight": 100,
    },
    45: {
        "fontName": "mplus1",
        "fontNameJa": "Mplus 1",
        "fontNameEn": "M PLUS 1",
        "fontWeight": 700,
    },
    46: {
        "fontName": "mplus1",
        "fontNameJa": "Mplus 1",
        "fontNameEn": "M PLUS 1",
        "fontWeight": 600,
    },
    47: {
        "fontName": "mplus1",
        "fontNameJa": "Mplus 1",
        "fontNameEn": "M PLUS 1",
        "fontWeight": 300,
    },
    48: {
        "fontName": "mplus1",
        "fontNameJa": "Mplus 1",
        "fontNameEn": "M PLUS 1",
        "fontWeight": 200,
    },
    49: {
        "fontName": "mplus2",
        "fontNameJa": "Mplus 2",
        "fontNameEn": "M PLUS 2",
        "fontWeight": 700,
    },
    50: {
        "fontName": "mplus2",
        "fontNameJa": "Mplus 2",
        "fontNameEn": "M PLUS 2",
        "fontWeight": 100,
    },
    51: {
        "fontName": "mplus2",
        "fontNameJa": "Mplus 2",
        "fontNameEn": "M PLUS 2",
        "fontWeight": 800,
    },
    52: {
        "fontName": "mplus2",
        "fontNameJa": "Mplus 2",
        "fontNameEn": "M PLUS 2",
        "fontWeight": 400,
    },
    53: {
        "fontName": "mplus2",
        "fontNameJa": "Mplus 2",
        "fontNameEn": "M PLUS 2",
        "fontWeight": 300,
    },
    54: {
        "fontName": "mplus2",
        "fontNameJa": "Mplus 2",
        "fontNameEn": "M PLUS 2",
        "fontWeight": 600,
    },
    55: {
        "fontName": "mplus2",
        "fontNameJa": "Mplus 2",
        "fontNameEn": "M PLUS 2",
        "fontWeight": 900,
    },
    56: {
        "fontName": "mplus2",
        "fontNameJa": "Mplus 2",
        "fontNameEn": "M PLUS 2",
        "fontWeight": 200,
    },
    57: {
        "fontName": "mplus2",
        "fontNameJa": "Mplus 2",
        "fontNameEn": "M PLUS 2",
        "fontWeight": 500,
    },
    58: {
        "fontName": "mplus1code",
        "fontNameJa": "Mplus 1 Code",
        "fontNameEn": "M PLUS 1 Code",
        "fontWeight": 100,
    },
    59: {
        "fontName": "mplus1code",
        "fontNameJa": "Mplus 1 Code",
        "fontNameEn": "M PLUS 1 Code",
        "fontWeight": 500,
    },
    60: {
        "fontName": "mplus1code",
        "fontNameJa": "Mplus 1 Code",
        "fontNameEn": "M PLUS 1 Code",
        "fontWeight": 700,
    },
    61: {
        "fontName": "mplus1code",
        "fontNameJa": "Mplus 1 Code",
        "fontNameEn": "M PLUS 1 Code",
        "fontWeight": 400,
    },
    62: {
        "fontName": "mplus1code",
        "fontNameJa": "Mplus 1 Code",
        "fontNameEn": "M PLUS 1 Code",
        "fontWeight": 600,
    },
    63: {
        "fontName": "mplus1code",
        "fontNameJa": "Mplus 1 Code",
        "fontNameEn": "M PLUS 1 Code",
        "fontWeight": 200,
    },
    64: {
        "fontName": "mplus1code",
        "fontNameJa": "Mplus 1 Code",
        "fontNameEn": "M PLUS 1 Code",
        "fontWeight": 300,
    },
    65: {
        "fontName": "notosansjp",
        "fontNameJa": "Noto Sans Japanese",
        "fontNameEn": "Noto Sans Japanese",
        "fontWeight": 400,
    },
    66: {
        "fontName": "notosansjp",
        "fontNameJa": "Noto Sans Japanese",
        "fontNameEn": "Noto Sans Japanese",
        "fontWeight": 500,
    },
    67: {
        "fontName": "notosansjp",
        "fontNameJa": "Noto Sans Japanese",
        "fontNameEn": "Noto Sans Japanese",
        "fontWeight": 300,
    },
    68: {
        "fontName": "notosansjp",
        "fontNameJa": "Noto Sans Japanese",
        "fontNameEn": "Noto Sans Japanese",
        "fontWeight": 100,
    },
    69: {
        "fontName": "notosansjp",
        "fontNameJa": "Noto Sans Japanese",
        "fontNameEn": "Noto Sans Japanese",
        "fontWeight": 700,
    },
    70: {
        "fontName": "notosansjp",
        "fontNameJa": "Noto Sans Japanese",
        "fontNameEn": "Noto Sans Japanese",
        "fontWeight": 900,
    },
    71: {
        "fontName": "notoserifjp",
        "fontNameJa": "Noto Serif Japanese",
        "fontNameEn": "Noto Serif Japanese",
        "fontWeight": 700,
    },
    72: {
        "fontName": "notoserifjp",
        "fontNameJa": "Noto Serif Japanese",
        "fontNameEn": "Noto Serif Japanese",
        "fontWeight": 500,
    },
    73: {
        "fontName": "notoserifjp",
        "fontNameJa": "Noto Serif Japanese",
        "fontNameEn": "Noto Serif Japanese",
        "fontWeight": 600,
    },
    74: {
        "fontName": "notoserifjp",
        "fontNameJa": "Noto Serif Japanese",
        "fontNameEn": "Noto Serif Japanese",
        "fontWeight": 300,
    },
    75: {
        "fontName": "notoserifjp",
        "fontNameJa": "Noto Serif Japanese",
        "fontNameEn": "Noto Serif Japanese",
        "fontWeight": 400,
    },
    76: {
        "fontName": "notoserifjp",
        "fontNameJa": "Noto Serif Japanese",
        "fontNameEn": "Noto Serif Japanese",
        "fontWeight": 900,
    },
    77: {
        "fontName": "notoserifjp",
        "fontNameJa": "Noto Serif Japanese",
        "fontNameEn": "Noto Serif Japanese",
        "fontWeight": 200,
    },
    78: {
        "fontName": "zenantiquesoft",
        "fontNameJa": "ZENアンチックソフト",
        "fontNameEn": "Zen Antique Soft",
        "fontWeight": 400,
    },
    79: {
        "fontName": "murecho",
        "fontNameJa": "Murecho",
        "fontNameEn": "Murecho",
        "fontWeight": 200,
    },
    80: {
        "fontName": "murecho",
        "fontNameJa": "Murecho",
        "fontNameEn": "Murecho",
        "fontWeight": 800,
    },
    81: {
        "fontName": "murecho",
        "fontNameJa": "Murecho",
        "fontNameEn": "Murecho",
        "fontWeight": 300,
    },
    82: {
        "fontName": "murecho",
        "fontNameJa": "Murecho",
        "fontNameEn": "Murecho",
        "fontWeight": 400,
    },
    83: {
        "fontName": "murecho",
        "fontNameJa": "Murecho",
        "fontNameEn": "Murecho",
        "fontWeight": 900,
    },
    84: {
        "fontName": "murecho",
        "fontNameJa": "Murecho",
        "fontNameEn": "Murecho",
        "fontWeight": 600,
    },
    85: {
        "fontName": "murecho",
        "fontNameJa": "Murecho",
        "fontNameEn": "Murecho",
        "fontWeight": 500,
    },
    86: {
        "fontName": "murecho",
        "fontNameJa": "Murecho",
        "fontNameEn": "Murecho",
        "fontWeight": 700,
    },
    87: {
        "fontName": "murecho",
        "fontNameJa": "Murecho",
        "fontNameEn": "Murecho",
        "fontWeight": 100,
    },
    88: {
        "fontName": "mochiypopone",
        "fontNameJa": "モッチーポップ One",
        "fontNameEn": "Mochiy Pop One",
        "fontWeight": 400,
    },
    89: {
        "fontName": "yujisyuku",
        "fontNameJa": "Yuji Syuku",
        "fontNameEn": "Yuji Syuku",
        "fontWeight": 400,
    },
    90: {
        "fontName": "yujiboku",
        "fontNameJa": "Yuji Boku",
        "fontNameEn": "Yuji Boku",
        "fontWeight": 400,
    },
    91: {
        "fontName": "yujimai",
        "fontNameJa": "Yuji Mai",
        "fontNameEn": "Yuji Mai",
        "fontWeight": 400,
    },
    92: {
        "fontName": "zenkakugothicnew",
        "fontNameJa": "ZEN角ゴシック",
        "fontNameEn": "Zen Kaku Gothic New",
        "fontWeight": 700,
    },
    93: {
        "fontName": "zenkakugothicnew",
        "fontNameJa": "ZEN角ゴシック",
        "fontNameEn": "Zen Kaku Gothic New",
        "fontWeight": 300,
    },
    94: {
        "fontName": "zenkakugothicnew",
        "fontNameJa": "ZEN角ゴシック",
        "fontNameEn": "Zen Kaku Gothic New",
        "fontWeight": 500,
    },
    95: {
        "fontName": "zenkakugothicnew",
        "fontNameJa": "ZEN角ゴシック",
        "fontNameEn": "Zen Kaku Gothic New",
        "fontWeight": 900,
    },
    96: {
        "fontName": "zenkakugothicnew",
        "fontNameJa": "ZEN角ゴシック",
        "fontNameEn": "Zen Kaku Gothic New",
        "fontWeight": 400,
    },
    97: {
        "fontName": "zenmarugothic",
        "fontNameJa": "ZEN丸ゴシック",
        "fontNameEn": "Zen Maru Gothic",
        "fontWeight": 700,
    },
    98: {
        "fontName": "zenmarugothic",
        "fontNameJa": "ZEN丸ゴシック",
        "fontNameEn": "Zen Maru Gothic",
        "fontWeight": 300,
    },
    99: {
        "fontName": "zenmarugothic",
        "fontNameJa": "ZEN丸ゴシック",
        "fontNameEn": "Zen Maru Gothic",
        "fontWeight": 500,
    },
    100: {
        "fontName": "zenmarugothic",
        "fontNameJa": "ZEN丸ゴシック",
        "fontNameEn": "Zen Maru Gothic",
        "fontWeight": 900,
    },
    101: {
        "fontName": "zenmarugothic",
        "fontNameJa": "ZEN丸ゴシック",
        "fontNameEn": "Zen Maru Gothic",
        "fontWeight": 400,
    },
    102: {
        "fontName": "zenkakugothicantique",
        "fontNameJa": "ZEN角ゴシックアンティーク",
        "fontNameEn": "Zen Kaku Gothic Antique",
        "fontWeight": 300,
    },
    103: {
        "fontName": "zenkakugothicantique",
        "fontNameJa": "ZEN角ゴシックアンティーク",
        "fontNameEn": "Zen Kaku Gothic Antique",
        "fontWeight": 700,
    },
    104: {
        "fontName": "zenkakugothicantique",
        "fontNameJa": "ZEN角ゴシックアンティーク",
        "fontNameEn": "Zen Kaku Gothic Antique",
        "fontWeight": 500,
    },
    105: {
        "fontName": "zenkakugothicantique",
        "fontNameJa": "ZEN角ゴシックアンティーク",
        "fontNameEn": "Zen Kaku Gothic Antique",
        "fontWeight": 400,
    },
    106: {
        "fontName": "zenkakugothicantique",
        "fontNameJa": "ZEN角ゴシックアンティーク",
        "fontNameEn": "Zen Kaku Gothic Antique",
        "fontWeight": 900,
    },
    107: {
        "fontName": "zenoldmincho",
        "fontNameJa": "ZENオールド明朝",
        "fontNameEn": "Zen Old Mincho",
        "fontWeight": 900,
    },
    108: {
        "fontName": "zenoldmincho",
        "fontNameJa": "ZENオールド明朝",
        "fontNameEn": "Zen Old Mincho",
        "fontWeight": 700,
    },
    109: {
        "fontName": "zenoldmincho",
        "fontNameJa": "ZENオールド明朝",
        "fontNameEn": "Zen Old Mincho",
        "fontWeight": 400,
    },
    110: {
        "fontName": "zenantique",
        "fontNameJa": "ZENアンティーク",
        "fontNameEn": "Zen Antique",
        "fontWeight": 400,
    },
    111: {
        "fontName": "zenkurenaido",
        "fontNameJa": "ZEN紅道",
        "fontNameEn": "Zen Kurenaido",
        "fontWeight": 400,
    },
    112: {
        "fontName": "shipporiantique",
        "fontNameJa": "しっぽりアンティーク",
        "fontNameEn": "Shippori Antique",
        "fontWeight": 400,
    },
    113: {
        "fontName": "morisawabizudgothic",
        "fontNameJa": "モリサワBIZ UDゴシック",
        "fontNameEn": "BIZ UDGothic",
        "fontWeight": 700,
    },
    114: {
        "fontName": "morisawabizudgothic",
        "fontNameJa": "モリサワBIZ UDゴシック",
        "fontNameEn": "BIZ UDGothic",
        "fontWeight": 400,
    },
    115: {
        "fontName": "morisawabizudmincho",
        "fontNameJa": "モリサワBIZ UD明朝",
        "fontNameEn": "BIZ UDMincho",
        "fontWeight": 400,
    },
    116: {
        "fontName": "morisawabizudpmincho",
        "fontNameJa": "モリサワBIZ UDP明朝",
        "fontNameEn": "BIZ UDPMincho",
        "fontWeight": 400,
    },
}
