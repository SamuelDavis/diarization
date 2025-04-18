(async () => {
  if (!targetName) throw new Error("target name undefined");
  console.countReset();
  window.calls = window.calls ?? {};
  const urlRoot = "https://voice.google.com/u/0/a/cr/";

  const scroller = document.querySelector(".cdk-virtual-scroll-viewport");
  scroller.scrollTop = 0;
  let scroll = scroller.scrollTop;
  let elements = Array.from(document.querySelectorAll("ol li"));
  for (let hardStop = 10; hardStop >= 0; hardStop--) {
    for (const element of elements) {
      const title = element.querySelector(".title").innerText;
      const [name = "", description = "", time = ""] = title
        .split(/[\n\.]/)
        .slice(0, 3)
        .map((s) => s.trim());

      if (name !== targetName) continue;
      if (!description.startsWith("Incoming")) continue;
      console.count();
      element.querySelector('[role="button"]').click();
      await new Promise((resolve) => setTimeout(resolve, 250));
      const searchParams = new URLSearchParams(location.search);
      const itemId = searchParams.get("itemId")?.slice(2);
      if (!itemId) throw new Error();
      const url = `${urlRoot}cra:${itemId}`;
      const date = new Date(time);
      window.calls[url] = date.getTime();
    }
    if (scroll === scroller.scrollTop) break;
    scroll = scroller.scrollTop;
    scroller.scrollTop += Math.floor(scroller.clientHeight / 3);
    elements = Array.from(document.querySelectorAll("ol li")).slice(9);
  }

  console.info(Object.entries(window.calls).sort(([, a], [, b]) => b - a));
})();
