NUM_TOP_K = 5
PATCH_SIZE = 105
NUM_PATCHES = 5
MODEL_PATH = "models/font77_vgg16.pt"

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
    15: {
        "fontName": "kleeone",
        "fontNameJa": "クレー One",
        "fontNameEn": "Klee One",
        "fontWeight": 400,
    },
    16: {
        "fontName": "kleeone",
        "fontNameJa": "クレー One",
        "fontNameEn": "Klee One",
        "fontWeight": 600,
    },
    17: {
        "fontName": "rampartone",
        "fontNameJa": "ランパート One",
        "fontNameEn": "Rampart One",
        "fontWeight": 400,
    },
    18: {
        "fontName": "shipporimincho",
        "fontNameJa": "しっぽり明朝",
        "fontNameEn": "Shippori Mincho",
        "fontWeight": 700,
    },
    19: {
        "fontName": "shipporimincho",
        "fontNameJa": "しっぽり明朝",
        "fontNameEn": "Shippori Mincho",
        "fontWeight": 600,
    },
    20: {
        "fontName": "shipporimincho",
        "fontNameJa": "しっぽり明朝",
        "fontNameEn": "Shippori Mincho",
        "fontWeight": 400,
    },
    21: {
        "fontName": "shipporimincho",
        "fontNameJa": "しっぽり明朝",
        "fontNameEn": "Shippori Mincho",
        "fontWeight": 500,
    },
    22: {
        "fontName": "shipporimincho",
        "fontNameJa": "しっぽり明朝",
        "fontNameEn": "Shippori Mincho",
        "fontWeight": 800,
    },
    23: {
        "fontName": "sawarabigothic",
        "fontNameJa": "さわらびゴシック",
        "fontNameEn": "Sawarabi Gothic",
        "fontWeight": 400,
    },
    24: {
        "fontName": "sawarabimincho",
        "fontNameJa": "さわらび明朝",
        "fontNameEn": "Sawarabi Mincho",
        "fontWeight": 400,
    },
    25: {
        "fontName": "newtegomin",
        "fontNameJa": "ニューテゴミン",
        "fontNameEn": "New Tegomin",
        "fontWeight": 400,
    },
    26: {
        "fontName": "kiwimaru",
        "fontNameJa": "キウイ丸",
        "fontNameEn": "Kiwi Maru",
        "fontWeight": 500,
    },
    27: {
        "fontName": "kiwimaru",
        "fontNameJa": "キウイ丸",
        "fontNameEn": "Kiwi Maru",
        "fontWeight": 300,
    },
    28: {
        "fontName": "kiwimaru",
        "fontNameJa": "キウイ丸",
        "fontNameEn": "Kiwi Maru",
        "fontWeight": 400,
    },
    29: {
        "fontName": "delagothicone",
        "fontNameJa": "デラゴシック",
        "fontNameEn": "Dela Gothic One",
        "fontWeight": 400,
    },
    30: {
        "fontName": "yomogi",
        "fontNameJa": "Yomogi",
        "fontNameEn": "Yomogi",
        "fontWeight": 400,
    },
    31: {
        "fontName": "hachimarupop",
        "fontNameJa": "はちまるポップ",
        "fontNameEn": "Hachi Maru Pop",
        "fontWeight": 400,
    },
    32: {
        "fontName": "pottaone",
        "fontNameJa": "ポッタ",
        "fontNameEn": "Potta One",
        "fontWeight": 400,
    },
    33: {
        "fontName": "stick",
        "fontNameJa": "ステッキ",
        "fontNameEn": "Stick",
        "fontWeight": 400,
    },
    34: {
        "fontName": "rocknrollone",
        "fontNameJa": "ロックンロール One",
        "fontNameEn": "RocknRoll One",
        "fontWeight": 400,
    },
    35: {
        "fontName": "reggaeone",
        "fontNameJa": "レゲエ One",
        "fontNameEn": "Reggae One",
        "fontWeight": 400,
    },
    36: {
        "fontName": "trainone",
        "fontNameJa": "トレイン One",
        "fontNameEn": "Train One",
        "fontWeight": 400,
    },
    37: {
        "fontName": "dotgothic16",
        "fontNameJa": "ドットゴシック16",
        "fontNameEn": "DotGothic16",
        "fontWeight": 400,
    },
    38: {
        "fontName": "yuseimagic",
        "fontNameJa": "YuseiMagic",
        "fontNameEn": "Yusei Magic",
        "fontWeight": 400,
    },
    39: {
        "fontName": "kosugi",
        "fontNameJa": "小杉フォント",
        "fontNameEn": "Kosugi",
        "fontWeight": 400,
    },
    40: {
        "fontName": "kosugimaru",
        "fontNameJa": "小杉丸フォント",
        "fontNameEn": "Kosugi Maru",
        "fontWeight": 400,
    },
    41: {
        "fontName": "mplus1",
        "fontNameJa": "Mplus 1",
        "fontNameEn": "M PLUS 1",
        "fontWeight": 400,
    },
    42: {
        "fontName": "mplus2",
        "fontNameJa": "Mplus 2",
        "fontNameEn": "M PLUS 2",
        "fontWeight": 400,
    },
    43: {
        "fontName": "mplus1code",
        "fontNameJa": "Mplus 1 Code",
        "fontNameEn": "M PLUS 1 Code",
        "fontWeight": 400,
    },
    44: {
        "fontName": "zenantiquesoft",
        "fontNameJa": "ZENアンチックソフト",
        "fontNameEn": "Zen Antique Soft",
        "fontWeight": 400,
    },
    45: {
        "fontName": "murecho",
        "fontNameJa": "Murecho",
        "fontNameEn": "Murecho",
        "fontWeight": 400,
    },
    46: {
        "fontName": "mochiypopone",
        "fontNameJa": "モッチーポップ One",
        "fontNameEn": "Mochiy Pop One",
        "fontWeight": 400,
    },
    47: {
        "fontName": "yujisyuku",
        "fontNameJa": "Yuji Syuku",
        "fontNameEn": "Yuji Syuku",
        "fontWeight": 400,
    },
    48: {
        "fontName": "yujiboku",
        "fontNameJa": "Yuji Boku",
        "fontNameEn": "Yuji Boku",
        "fontWeight": 400,
    },
    49: {
        "fontName": "yujimai",
        "fontNameJa": "Yuji Mai",
        "fontNameEn": "Yuji Mai",
        "fontWeight": 400,
    },
    50: {
        "fontName": "zenkakugothicnew",
        "fontNameJa": "ZEN角ゴシック",
        "fontNameEn": "Zen Kaku Gothic New",
        "fontWeight": 700,
    },
    51: {
        "fontName": "zenkakugothicnew",
        "fontNameJa": "ZEN角ゴシック",
        "fontNameEn": "Zen Kaku Gothic New",
        "fontWeight": 300,
    },
    52: {
        "fontName": "zenkakugothicnew",
        "fontNameJa": "ZEN角ゴシック",
        "fontNameEn": "Zen Kaku Gothic New",
        "fontWeight": 500,
    },
    53: {
        "fontName": "zenkakugothicnew",
        "fontNameJa": "ZEN角ゴシック",
        "fontNameEn": "Zen Kaku Gothic New",
        "fontWeight": 900,
    },
    54: {
        "fontName": "zenkakugothicnew",
        "fontNameJa": "ZEN角ゴシック",
        "fontNameEn": "Zen Kaku Gothic New",
        "fontWeight": 400,
    },
    55: {
        "fontName": "zenmarugothic",
        "fontNameJa": "ZEN丸ゴシック",
        "fontNameEn": "Zen Maru Gothic",
        "fontWeight": 700,
    },
    56: {
        "fontName": "zenmarugothic",
        "fontNameJa": "ZEN丸ゴシック",
        "fontNameEn": "Zen Maru Gothic",
        "fontWeight": 300,
    },
    57: {
        "fontName": "zenmarugothic",
        "fontNameJa": "ZEN丸ゴシック",
        "fontNameEn": "Zen Maru Gothic",
        "fontWeight": 500,
    },
    58: {
        "fontName": "zenmarugothic",
        "fontNameJa": "ZEN丸ゴシック",
        "fontNameEn": "Zen Maru Gothic",
        "fontWeight": 900,
    },
    59: {
        "fontName": "zenmarugothic",
        "fontNameJa": "ZEN丸ゴシック",
        "fontNameEn": "Zen Maru Gothic",
        "fontWeight": 400,
    },
    60: {
        "fontName": "zenkakugothicantique",
        "fontNameJa": "ZEN角ゴシックアンティーク",
        "fontNameEn": "Zen Kaku Gothic Antique",
        "fontWeight": 300,
    },
    61: {
        "fontName": "zenkakugothicantique",
        "fontNameJa": "ZEN角ゴシックアンティーク",
        "fontNameEn": "Zen Kaku Gothic Antique",
        "fontWeight": 700,
    },
    62: {
        "fontName": "zenkakugothicantique",
        "fontNameJa": "ZEN角ゴシックアンティーク",
        "fontNameEn": "Zen Kaku Gothic Antique",
        "fontWeight": 500,
    },
    63: {
        "fontName": "zenkakugothicantique",
        "fontNameJa": "ZEN角ゴシックアンティーク",
        "fontNameEn": "Zen Kaku Gothic Antique",
        "fontWeight": 400,
    },
    64: {
        "fontName": "zenkakugothicantique",
        "fontNameJa": "ZEN角ゴシックアンティーク",
        "fontNameEn": "Zen Kaku Gothic Antique",
        "fontWeight": 900,
    },
    65: {
        "fontName": "zenoldmincho",
        "fontNameJa": "ZENオールド明朝",
        "fontNameEn": "Zen Old Mincho",
        "fontWeight": 600,
    },
    66: {
        "fontName": "zenoldmincho",
        "fontNameJa": "ZENオールド明朝",
        "fontNameEn": "Zen Old Mincho",
        "fontWeight": 900,
    },
    67: {
        "fontName": "zenoldmincho",
        "fontNameJa": "ZENオールド明朝",
        "fontNameEn": "Zen Old Mincho",
        "fontWeight": 700,
    },
    68: {
        "fontName": "zenoldmincho",
        "fontNameJa": "ZENオールド明朝",
        "fontNameEn": "Zen Old Mincho",
        "fontWeight": 400,
    },
    69: {
        "fontName": "zenoldmincho",
        "fontNameJa": "ZENオールド明朝",
        "fontNameEn": "Zen Old Mincho",
        "fontWeight": 500,
    },
    70: {
        "fontName": "zenantique",
        "fontNameJa": "ZENアンティーク",
        "fontNameEn": "Zen Antique",
        "fontWeight": 400,
    },
    71: {
        "fontName": "zenkurenaido",
        "fontNameJa": "ZEN紅道",
        "fontNameEn": "Zen Kurenaido",
        "fontWeight": 400,
    },
    72: {
        "fontName": "shipporiantique",
        "fontNameJa": "しっぽりアンティーク",
        "fontNameEn": "Shippori Antique",
        "fontWeight": 400,
    },
    14: {
        "fontName": "OtomanopeeOne",
        "fontNameJa": "OtomanopeeOne",
        "fontNameEn": "OtomanopeeOne",
        "fontWeight": 400,
    },
    73: {
        "fontName": "morisawabizudgothic",
        "fontNameJa": "モリサワBIZ UDゴシック",
        "fontNameEn": "BIZ UDGothic",
        "fontWeight": 700,
    },
    74: {
        "fontName": "morisawabizudmincho",
        "fontNameJa": "モリサワBIZ UD明朝",
        "fontNameEn": "BIZ UDGothic",
        "fontWeight": 400,
    },
    75: {
        "fontName": "morisawabizudgothic",
        "fontNameJa": "モリサワBIZ UDゴシック",
        "fontNameEn": "BIZ UDMincho",
        "fontWeight": 400,
    },
    76: {
        "fontName": "udp mincho",
        "fontNameJa": "udp mincho",
        "fontNameEn": "udp mincho",
        "fontWeight": 400,
    },
}
