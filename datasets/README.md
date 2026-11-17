# Where to put your training data

## 1. Sensor data (for crop recommendation) -> `datasets/sensor_data/`

Put a CSV file here with these exact column names:

    N, P, K, temperature, humidity, moisture, label

`label` is the crop name (e.g. rice, wheat, maize). A synthetic sample file
`crop_recommendation.csv` is already included so you can run the training
pipeline immediately as a smoke test — replace it with your real dataset
(e.g. the public "Crop Recommendation Dataset" on Kaggle) before relying on
predictions in production. If your real dataset has extra columns (rainfall,
ph, etc.) either drop them or extend `FEATURE_COLUMNS` in
`training/train_crop_model.py`.

## 2. Crop growth-stage images -> `datasets/crop_images/<crop>/stage/<stage_name>/`

Folder names must exactly match the configured crops and stages in
`config/app_settings.json`:

    datasets/crop_images/rice/stage/seedling/*.jpg
    datasets/crop_images/rice/stage/tillering/*.jpg
    datasets/crop_images/rice/stage/vegetative/*.jpg
    datasets/crop_images/rice/stage/flowering/*.jpg
    datasets/crop_images/rice/stage/maturity/*.jpg

    datasets/crop_images/wheat/stage/<same five folders>/*.jpg
    datasets/crop_images/maize/stage/<same five folders>/*.jpg

Aim for at least 100-200 images per stage per crop for a usable model; more
is better. Mixed JPG/PNG is fine.

## 3. Disease images -> `datasets/crop_images/disease/<crop>/<disease_class>/`

    datasets/crop_images/disease/rice/rice_blast/*.jpg
    datasets/crop_images/disease/rice/rice_brown_spot/*.jpg
    datasets/crop_images/disease/rice/healthy/*.jpg

    datasets/crop_images/disease/wheat/wheat_rust/*.jpg
    datasets/crop_images/disease/wheat/wheat_powdery_mildew/*.jpg
    datasets/crop_images/disease/wheat/healthy/*.jpg

    datasets/crop_images/disease/maize/maize_leaf_blight/*.jpg
    datasets/crop_images/disease/maize/maize_rust/*.jpg
    datasets/crop_images/disease/maize/healthy/*.jpg

IMPORTANT: every `<disease_class>` folder name must have a matching entry in
`config/disease_treatment.json`, since that file is what supplies the
treatment/severity/prevention text returned by the API. If you train on a
new disease class, add a corresponding entry to that JSON file (no code
changes needed).

You do not have to use exactly these disease names — they're examples. Just
keep the JSON config and your folder names in sync.
