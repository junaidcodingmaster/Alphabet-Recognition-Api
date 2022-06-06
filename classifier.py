try:
    import requests
    import numpy as np
    import pandas as pd
    from PIL import Image
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression

except AttributeError:
    pass
except ModuleNotFoundError:
    pass
else:
    pass

print("\nProgram Started !\n")

X = np.load("image.npz")["arr_0"]
y = pd.read_csv("labels.csv")["labels"]

# print(pd.Series(y).value_counts())
print("\nProgram Working ...\n")

classes = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]
nclasses = len(classes)

print("\nProcessing Inbuilt Data ...\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=9, train_size=3500, test_size=500
)
X_train_scaled = X_train / 255.0
X_test_scaled = X_test / 255.0

clf = LogisticRegression(solver="saga", multi_class="multinomial").fit(
    X_train_scaled, y_train
)

print("\nInbuilt Data Processed !\n")


def getPrediction(image):
    print("\nData Incoming ...\n")

    link = (
        "https://i1.wp.com/storage.googleapis.com/expo-website-default-avatars/"
        + image
        + ".png"
    )

    print("\nData Being Ready to process ...\n")

    img = requests.get(link)
    path = "./" + image + ".png"

    with open(path, "wb") as f:
        f.write(img.content)
        print("\nData is Ready to process !\n")
        print(path)
        
    print("\nLoading ...\n")

    print("\nProcessing Letter Prediction ...\n")

    im_pil = Image.open(path)
    image_bw = im_pil.convert("L")

    print("\nFitting Letter to every Letter ...\n")

    image_bw_resized = image_bw.resize((22, 30), Image.ANTIALIAS)
    pixel_filter = 20

    min_pixel = np.percentile(image_bw_resized, pixel_filter)
    image_bw_resized_inverted_scaled = np.clip(image_bw_resized - min_pixel, 0, 255)

    max_pixel = np.max(image_bw_resized)
    image_bw_resized_inverted_scaled = (
        np.asarray(image_bw_resized_inverted_scaled) / max_pixel
    )

    print("\nCalculating Prediction ...\n")

    test_sample = np.array(image_bw_resized_inverted_scaled).reshape(1, 660)
    test_pred = clf.predict(test_sample)

    print("\nCalculated Prediction !\n")

    return str(test_pred[0])

