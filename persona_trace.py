from transformers import GPT2Tokenizer, GPT2LMHeadModel
#from transformers import AutoTokenizer, AutoModel
import torch

#load the model from online hugging face library
tokenizer = GPT2Tokenizer.from_pretrained("af1tang/personaGPT")
model = GPT2LMHeadModel.from_pretrained("af1tang/personaGPT")
if torch.cuda.is_available():
    model = model.cuda()
## utility functions ##
flatten = lambda l: [item for sublist in l for item in sublist]
def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.data.numpy()

def to_var(x):
    if not torch.is_tensor(x):
        x = torch.Tensor(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def display_dialog_history(dialog_hx):
    for j, line in enumerate(dialog_hx):
        msg = tokenizer.decode(line)
        if j %2 == 0:
            print(">> User: "+ msg)
        else:
            print("Bot: "+msg)
            print()

def generate_next(bot_input_ids, do_sample=True, top_k=10, top_p=.92,
                  max_length=1000, pad_token=tokenizer.eos_token_id):
    full_msg = model.generate(bot_input_ids, do_sample=True,
                                              top_k=top_k, top_p=top_p, 
                                              max_length=max_length, pad_token_id=tokenizer.eos_token_id)
    msg = to_data(full_msg.detach()[0])[bot_input_ids.shape[-1]:]
    return msg

# get personality facts for conversation
facts = ["I love cooking", "I want to visit Mars", "I hate Mondays"]
personas = []
for i in range(3):
    response = facts[i] + tokenizer.eos_token
    personas.append(response)
personas = tokenizer.encode(''.join(['<|p2|>'] + personas + ['<|sep|>'] + ['<|start|>']))

## available actions ##
action_space = [ 'ask about kids.', "ask about pets.", 'talk about work.',
               'ask about marital status.', 'talk about travel.', 'ask about age and gender.',
        'ask about hobbies.', 'ask about favorite food.', 'talk about movies.',
        'talk about music.', 'talk about politics.']
# converse for 8 turns
dialog_hx = []
for step in range(11):
    # choose an action
    act = action_space[step]
    # format into prefix code
    action_prefix = tokenizer.encode(''.join(['<|act|> '] + [act] + ['<|p1|>'] + [] + ['<|sep|>'] + ['<|start|>']))
    bot_input_ids = to_var([action_prefix + flatten(dialog_hx)]).long()

    # generate query conditioned on action
    msg = generate_next(bot_input_ids)
    dialog_hx.append(msg)

    # generate bot response
    bot_input_ids = to_var([personas+ flatten(dialog_hx)]).long()
    msg = generate_next(bot_input_ids)
    dialog_hx.append(msg)
display_dialog_history(dialog_hx)

