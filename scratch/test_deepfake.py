import os
from urllib.request import urlretrieve
from src.intelligence.deepfake_detector import detect_deepfake_image

# Let's download a known real image and known fake image from standard datasets or wikipedia
real_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/React-icon.svg/200px-React-icon.svg.png"
# For a fake image, we can try to use a known AI image URL from wikimedia or just test the model on the real one.
fake_url = "https://upload.wikimedia.org/wikipedia/commons/2/23/Ai-generated-8314150_1280.jpg" 

os.makedirs("scratch", exist_ok=True)
real_path = "scratch/test_real.png"
fake_path = "scratch/test_fake.jpg"

print("Downloading images...")
urlretrieve(real_url, real_path)
urlretrieve(fake_url, fake_path)

print("\n--- Testing Real Image ---")
res1 = detect_deepfake_image(real_path)
print(res1)

print("\n--- Testing Fake Image ---")
res2 = detect_deepfake_image(fake_path)
print(res2)
