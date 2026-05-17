"""Config — edit settings.yaml / strategy_params.yaml / portfolio.yaml with backup."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st
from streamlit_ace import st_ace

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dashboard.lib import config_editor, state  # noqa: E402

st.set_page_config(page_title="Config", page_icon="⚙️", layout="wide")
st.title("⚙️ Config editor")

configs = config_editor.list_yaml_configs()
if not configs:
    st.warning(f"No YAML files under {state.CONFIG_DIR}.")
    st.stop()

with st.sidebar:
    st.markdown("### File")
    labels = [str(p.relative_to(state.REPO_ROOT)) for p in configs]
    idx = st.selectbox("Config file", options=list(range(len(configs))),
                       format_func=lambda i: labels[i])
    selected = configs[idx]
    st.markdown("### Editor")
    theme = st.selectbox("Theme", ["github", "monokai", "tomorrow", "twilight",
                                     "tomorrow_night", "solarized_light"], index=0)
    font_size = st.slider("Font size", min_value=10, max_value=20, value=13)
    auto_update = st.checkbox("Live diff on every keystroke", value=False,
                              help="If off, press Apply / Refresh diff to update.")

st.caption(f"Editing: `{selected}` · {selected.stat().st_size} bytes")

current_text = config_editor.read_text(selected)

if "config_buffer" not in st.session_state or \
   st.session_state.get("config_path") != str(selected):
    st.session_state.config_buffer = current_text
    st.session_state.config_path = str(selected)

new_text = st_ace(
    value=st.session_state.config_buffer,
    language="yaml",
    theme=theme,
    font_size=font_size,
    key=f"ace_{selected.name}",
    height=480,
    auto_update=auto_update,
    show_gutter=True,
    wrap=True,
)

if new_text is not None:
    st.session_state.config_buffer = new_text


# ---------------- validate ----------------
ok, err, parsed = config_editor.validate_yaml(st.session_state.config_buffer)
if ok:
    st.success("YAML is valid.")
else:
    st.error(f"YAML invalid: {err}")

# ---------------- diff ----------------
diff = config_editor.diff_text(current_text, st.session_state.config_buffer,
                                 path=selected.name)
if not diff:
    st.info("No changes vs file on disk.")
else:
    with st.expander("Diff vs disk", expanded=True):
        st.code(diff, language="diff")

# ---------------- apply ----------------
st.markdown("---")
a1, a2, a3, a4 = st.columns(4)
with a1:
    apply = st.button("💾 Apply", disabled=not ok or not diff,
                      type="primary", use_container_width=True)
with a2:
    reload_disk = st.button("🔄 Reload from disk", use_container_width=True)
with a3:
    if st.button("↩️ Restore from latest backup", use_container_width=True):
        backups = sorted((state.CONFIG_DIR / ".backups").glob(f"{selected.name}.*"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        if not backups:
            st.error(f"No backup found for {selected.name}.")
        else:
            st.session_state.config_buffer = backups[0].read_text()
            st.success(f"Loaded {backups[0].name} into editor.")
            st.rerun()
with a4:
    open_backups = st.button("📦 Browse backups", use_container_width=True)


if apply:
    try:
        bp = config_editor.save(selected, st.session_state.config_buffer)
        st.success(f"Saved! Backup written to `{bp}`.")
        st.cache_data.clear()
        st.rerun()
    except ValueError as e:
        st.error(str(e))

if reload_disk:
    st.session_state.config_buffer = current_text
    st.rerun()

if open_backups:
    backups = sorted((state.CONFIG_DIR / ".backups").glob(f"{selected.name}.*"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
    if not backups:
        st.info(f"No backups yet for {selected.name}.")
    else:
        st.markdown("### Backups")
        for b in backups[:30]:
            with st.expander(b.name):
                st.code(b.read_text()[:8000], language="yaml")
