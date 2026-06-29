import splitfolders

crops = [
    "corn",
    "groundnut",
    "pearlmillet",
    "rice",
    "soybean",
    "wheat"
]

for crop in crops:
    splitfolders.ratio(
        input=f"data/{crop}",
        output=f"data_split/{crop}",
        seed=42,
        ratio=(0.7, 0.15, 0.15)
    )