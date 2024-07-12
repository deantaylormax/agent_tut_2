#here we can create any function of any complexity that can then be run by our model
from llama_index.core.tools import FunctionTool

note_file = 'data/notes.txt'

def save_note(note):
    if not note_file:
        open(note_file, 'w')
        
    with open(note_file, 'a') as f:
        f.writelines([note + "\n"])
        
    return "note saved"

note_engine = FunctionTool.from_defaults(
    fn=save_note,
    name='note_saver',
    description="Save a text based note created by the user to a file",  #this description helps the LLM understand what the function does and to better choose which tools to use based upon the user interactions
)