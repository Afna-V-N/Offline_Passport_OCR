from paddleocr import PaddleOCR
import time

IMAGE_PATH = "./test_tilt/passport2_tilt.jpeg"

print("Loading PaddleOCR...")

ocr = PaddleOCR(
    lang="en",
    device="cpu",
    enable_mkldnn=False
)

print("PaddleOCR loaded.")

start = time.perf_counter()

result = ocr.predict(
    IMAGE_PATH
)

elapsed = (time.perf_counter() - start) * 1000

print("\n" + "=" * 60)
print("PADDLEOCR RESULT")
print("=" * 60)

for res in result:
    print(res)

print("=" * 60)
print(f"Processing time: {elapsed:.0f} ms")
print("=" * 60)