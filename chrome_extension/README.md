# Chrome Extension

This folder contains a Chrome extension that analyzes the current tab URL using
the local phishing detection web app API.

## What it does

- Reads the active tab URL.
- Sends the URL to the local detection API.
- Shows threat level, risk score, confidence score, reasoning, and recommended
  action.
- Stores the latest detection result in Chrome local storage.
- Records extension scans in the web app logs.

The extension does not contain a separate model. It uses the same detection
engine as the web app so results stay consistent.

## Required local service

Start the web app before using the extension:

From the repository root:
cd .\web_app\
python app.py

Open:
http://127.0.0.1:5000

## Load in Chrome developer mode

1. Open Chrome.
2. Go to:
   chrome://extensions

3. Enable **Developer mode**.
4. Click **Load unpacked**.
5. Select the `chrome_extension/` folder.
6. Pin **Phishing URL Intelligence** from the extensions menu if desired.

## Use the extension

1. Open any HTTP or HTTPS page.
2. Click the extension icon.
3. Confirm the current URL shown in the popup.
4. Click **Analyze Current Page**.
5. Review threat level, risk score, confidence, reasons, and recommended action.

## Input/output flow

```text
Current browser tab URL
  |
  v
Extension popup
  |
  v
POST http://127.0.0.1:5000/api/analyze
  |
  v
Shared phishing detection engine
  |
  v
Threat level + risk score + confidence + reasoning
  |
  v
Extension popup result
```

## Troubleshooting

- If the popup says the detection service is unavailable, start the web app and
  try again.
- Chrome blocks analysis for internal pages such as `chrome://extensions`.
  Test with normal `http://` or `https://` pages.
- Retrain the model from the web app when you want the extension to use a new
  uploaded CSV dataset.
- Open the web app Logs & History section to review extension scan records.
