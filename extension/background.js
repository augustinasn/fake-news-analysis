
var url = "localhost:5000/";

function openMyPage() {
	browser.tabs.create({
		"url": url
	});
}

browser.browserAction.onClicked.addListener(openMyPage);

