chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.local.set({
    apiBaseUrl: "http://127.0.0.1:5000",
    lastDetection: null
  });
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === "GET_ACTIVE_TAB_URL") {
    sendResponse({ url: sender.tab?.url || "" });
  }
  return true;
});
