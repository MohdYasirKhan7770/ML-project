from predictor import predict_news

def run_tests():
    print("==================================================")
    print("           MODEL PREDICTION TESTS                 ")
    print("==================================================")

    # 1. Known REAL news samples
    real_news = [
        "President Joe Biden signed a $1.2 trillion bipartisan infrastructure bill into law on Monday, an event that brought together Democrats and Republicans on the White House lawn.",
        "The Dow Jones Industrial Average fell by 300 points on Tuesday amid concerns over rising inflation and supply chain disruptions affecting tech stocks.",
        "NASA's James Webb Space Telescope has successfully unfolded its final primary mirror segment, completing a major deployment milestone before it begins capturing images of the early universe.",
        "The World Health Organization announced today that it is monitoring a new variant of the COVID-19 virus, urging countries to maintain their sequencing efforts."
    ]

    print("\n--- TESTING KNOWN REAL NEWS ---")
    for i, text in enumerate(real_news, 1):
        result = predict_news(text)
        status = "✅ PASS" if result['raw_prediction'] == 1 else "❌ FAIL"
        print(f"\nTest {i}: {status} (Expected: REAL)")
        print(f"Prediction: {result['label']} (Confidence: {result['confidence']:.2%})")
        print(f"Excerpt: {text[:80]}...")

    # 2. Known FAKE news samples
    fake_news = [
        "BREAKING: Pope Francis Shocks World, Endorses Donald Trump for President, Releases Statement",
        "Scientists Discover Secret Underground Alien Base in Antarctica, Government Hiding Truth from Public!",
        "Eating raw garlic every morning cures all forms of cancer and prevents COVID-19 perfectly, claims anonymous doctor.",
        "New Law Passed: All Citizens Must Now Microchip Their Pets for Constant Government Surveillance Operations."
    ]

    print("\n\n--- TESTING KNOWN FAKE NEWS ---")
    for i, text in enumerate(fake_news, 1):
        result = predict_news(text)
        status = "✅ PASS" if result['raw_prediction'] == 0 else "❌ FAIL"
        print(f"\nTest {i}: {status} (Expected: FAKE)")
        print(f"Prediction: {result['label']} (Confidence: {result['confidence']:.2%})")
        print(f"Excerpt: {text[:80]}...")

if __name__ == '__main__':
    run_tests()
