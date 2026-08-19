from modules import script_callbacks

from anima_layer_lab.ui import create_ui


script_callbacks.on_ui_tabs(create_ui, name="anima_layer_lab")
