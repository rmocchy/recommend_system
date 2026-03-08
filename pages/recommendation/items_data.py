"""Recommendation System — Sample product catalog definitions."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Item:
    id: int
    name: str
    category: str
    price: int        # price
    score: float      # rating score from 0.0 to 5.0
    emoji: str        # product icon
    description: str  # short product description


# ── Category color mapping ────────────────────────────────────
CATEGORY_COLOR: dict[str, str] = {
    "Phones & PCs":  "#4A90D9",
    "Audio":         "#7B68EE",
    "Wearables":     "#5BA85C",
    "Books":         "#E8963C",
    "Fashion":       "#E05C8A",
    "Food":          "#D4A017",
    "Sports":        "#2AABB5",
    "Home":          "#A05CB0",
}

CATEGORY_BG: dict[str, str] = {
    "Phones & PCs":  "#E8F0FA",
    "Audio":         "#EEEAFF",
    "Wearables":     "#E8F5E9",
    "Books":         "#FFF3E0",
    "Fashion":       "#FCE4EC",
    "Food":          "#FFFDE7",
    "Sports":        "#E0F7FA",
    "Home":          "#F3E5F5",
}


# ── Default product catalog ───────────────────────────────────
DEFAULT_ITEMS: list[Item] = [
    # Phones & PCs
    Item(0,  "Ultra Laptop Pro 16",      "Phones & PCs", 29800, 4.7, "💻", "Powerful M3 chip, 16-inch Retina display"),
    Item(1,  "Smartphone Z Pro",         "Phones & PCs", 24800, 4.5, "📱", "120Hz AMOLED, 200MP camera flagship"),
    Item(2,  "Tablet Air 11",            "Phones & PCs", 18800, 4.3, "📟", "Lightweight 450g, up to 12-hour battery"),
    # Audio
    Item(3,  "Wireless Earphones X",     "Audio",        14800, 4.6, "🎧", "Active noise canceling, spatial audio, 30-hour playback"),
    Item(4,  "Studio Monitor HP",        "Audio",        22800, 4.4, "🎵", "Hi-Res Audio, professional-grade headphones"),
    Item(5,  "Bluetooth Speaker",        "Audio",         9800, 4.2, "🔊", "Waterproof IP67, 360° sound, 20W output"),
    # Wearables
    Item(6,  "Smartwatch Ultra",         "Wearables",    27800, 4.8, "⌚", "ECG & blood oxygen sensor, GPS, titanium case"),
    Item(7,  "Fitness Band",             "Wearables",     8800, 4.0, "🏃", "24h heart rate monitor, sleep score, 7-day battery"),
    # Books
    Item(8,  "Learn Python 3rd Edition", "Books",         3500, 4.5, "📖", "For beginners to intermediate learners, 300 exercises included"),
    Item(9,  "Intro to Quantum Computing","Books",        4200, 4.3, "🔬", "Covers QUBO and annealing — the latest in quantum tech"),
    Item(10, "World Cuisine Encyclopedia","Books",        3200, 4.1, "🍽️", "500 recipes from 50 countries, full-color photos"),
    # Fashion
    Item(11, "Premium Sneakers",         "Fashion",      16800, 4.4, "👟", "Genuine leather upper, cushioned sole, 3 colors available"),
    Item(12, "Cashmere Knit",            "Fashion",      19800, 4.6, "🧥", "100% Mongolian cashmere, machine washable"),
    Item(13, "Leather Tote Bag",         "Fashion",      25800, 4.2, "👜", "Fits A4 size, genuine leather handles, magnetic closure"),
    # Food
    Item(14, "Specialty Coffee",         "Food",          4500, 4.7, "☕", "Ethiopian single-origin, 200g"),
    Item(15, "Protein Bars (12 pack)",   "Food",          3200, 4.0, "🍫", "20g protein, no added sugar, 6 flavors"),
    Item(16, "Organic Green Tea Set",    "Food",          3000, 4.3, "🍵", "Premium first-harvest green tea, 5-variety assortment"),
    # Sports
    Item(17, "Yoga Mat Premium",         "Sports",        8900, 4.5, "🧘", "6mm thick, non-slip surface, carrying bag included"),
    Item(18, "Adjustable Dumbbell 40kg", "Sports",       28800, 4.4, "🏋️", "Adjustable in 2.5kg steps, compact storage, rack included"),
    # Home
    Item(19, "HEPA Air Purifier",        "Home",         26800, 4.6, "🌬️", "True HEPA + activated carbon filter, covers up to 66 m²"),
    Item(20, "Aroma Diffuser",           "Home",          5800, 4.3, "🕯️", "Ultrasonic, 7-color LED, timer included"),
]

ALL_CATEGORIES: list[str] = sorted({item.category for item in DEFAULT_ITEMS})
