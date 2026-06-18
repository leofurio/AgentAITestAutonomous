"""Build a compact, ref-indexed outline of the page from the live DOM.

Each interactive/visible element is tagged with a stable ``data-aiwebtest-ref``
attribute so the agent can address it by ref id (e.g. ``e12``) instead of a brittle
CSS selector. The locator for a ref is ``[data-aiwebtest-ref="<ref>"]``.
"""

from __future__ import annotations

from dataclasses import dataclass

# JS that walks the DOM, tags interactive/meaningful elements with a ref attribute,
# and returns a flat list of {ref, tag, role, name, value}. Runs in the page context.
_SNAPSHOT_JS = r"""
() => {
  const REF_ATTR = 'data-aiwebtest-ref';
  const INTERACTIVE = new Set(['A', 'BUTTON', 'INPUT', 'SELECT', 'TEXTAREA']);
  const elements = [];
  let counter = 0;

  // Clear refs from a previous snapshot first: the DOM changes between snapshots, and
  // leaving stale attributes makes a ref id (e.g. e2) match several elements, which breaks
  // ref-based locators with a strict-mode "resolved to N elements" error.
  document.querySelectorAll('[' + REF_ATTR + ']').forEach((el) => el.removeAttribute(REF_ATTR));

  const isVisible = (el) => {
    const style = window.getComputedStyle(el);
    if (style.display === 'none' || style.visibility === 'hidden' || style.opacity === '0') {
      return false;
    }
    const rect = el.getBoundingClientRect();
    return rect.width > 0 && rect.height > 0;
  };

  const accessibleName = (el) => {
    const aria = el.getAttribute('aria-label');
    if (aria) return aria.trim();
    if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT') {
      if (el.labels && el.labels.length) return el.labels[0].textContent.trim();
      const ph = el.getAttribute('placeholder');
      if (ph) return ph.trim();
      const name = el.getAttribute('name');
      if (name) return name.trim();
    }
    const text = (el.innerText || el.textContent || '').trim().replace(/\s+/g, ' ');
    return text.slice(0, 120);
  };

  const role = (el) => {
    const explicit = el.getAttribute('role');
    if (explicit) return explicit;
    const tag = el.tagName;
    if (tag === 'A') return 'link';
    if (tag === 'BUTTON') return 'button';
    if (tag === 'SELECT') return 'select';
    if (tag === 'TEXTAREA') return 'textbox';
    if (tag === 'INPUT') return (el.getAttribute('type') || 'text');
    if (/^H[1-6]$/.test(tag)) return 'heading';
    return tag.toLowerCase();
  };

  const candidates = document.querySelectorAll(
    'a, button, input, select, textarea, [role="button"], [role="link"], [onclick], ' +
    'h1, h2, h3, label, [data-testid]'
  );

  for (const el of candidates) {
    if (!isVisible(el)) continue;
    const name = accessibleName(el);
    const interactive = INTERACTIVE.has(el.tagName) ||
      el.hasAttribute('onclick') || ['button', 'link'].includes(el.getAttribute('role'));
    if (!name && !interactive) continue;
    const ref = 'e' + (++counter);
    el.setAttribute(REF_ATTR, ref);
    elements.push({
      ref,
      tag: el.tagName.toLowerCase(),
      role: role(el),
      name,
      value: (el.value !== undefined ? String(el.value) : '').slice(0, 80),
      el_id: el.id || '',
      attr_name: el.getAttribute('name') || '',
      testid: el.getAttribute('data-testid') || '',
    });
  }
  return elements;
};
"""


@dataclass
class SnapshotElement:
    ref: str
    tag: str
    role: str
    name: str
    value: str
    el_id: str = ""
    attr_name: str = ""
    testid: str = ""

    def locator_hint(self) -> dict[str, str]:
        """A stable descriptor used to generate a robust replay locator."""
        return {
            "id": self.el_id,
            "testid": self.testid,
            "attr_name": self.attr_name,
            "role": self.role,
            "name": self.name,
            "tag": self.tag,
        }


def ref_selector(ref: str) -> str:
    """CSS selector that resolves a snapshot ref id to its element."""
    return f'[data-aiwebtest-ref="{ref}"]'


async def take_snapshot(page) -> tuple[str, list[SnapshotElement]]:
    """Return a (text outline, elements) tuple for the current page state."""
    raw = await page.evaluate(_SNAPSHOT_JS)
    elements = [SnapshotElement(**item) for item in raw]
    return render_outline(page.url, elements), elements


def render_outline(url: str, elements: list[SnapshotElement]) -> str:
    """Render elements into a compact, token-efficient text outline."""
    lines = [f"URL: {url}", f"Interactive/visible elements ({len(elements)}):"]
    for el in elements:
        parts = [f"[ref={el.ref}]", el.role]
        if el.name:
            parts.append(f'"{el.name}"')
        if el.value:
            parts.append(f"(value={el.value!r})")
        lines.append("  " + " ".join(parts))
    if not elements:
        lines.append("  (no interactive elements detected)")
    return "\n".join(lines)
