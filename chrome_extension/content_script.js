chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "CURRENT_PAGE_CONTEXT") {
    sendResponse({
      url: window.location.href,
      title: document.title
    });
  }
  return true;
});
