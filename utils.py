from PIL import Image, ImageDraw, ImageFont

def str2bool(v):
    """Convert string to boolean"""
    return v.lower().strip() == 'true'

def visualize_array(arr):
    """Helper to generate array visualization"""

    src_img = Image.new('RGB', (110, 200), 'black')
    draw = ImageDraw.Draw(src_img)
    
    for i, element in enumerate(arr):
        draw.rectangle(
            ((20*i+10, 200), (20*i + 20, 200 - 200*(element+1)/5)),
            fill='white'
        )

    return src_img


def visualize_agents(agents, agent_names):
    """Helper to generate agents in a row visualization"""
    
    n = len(agents)
    src_img = Image.new('RGB', (200*n, 300), 'black')
    
    for i, agent in enumerate(agents):
        src_img.paste(visualize_array(agent.arr), (200*i + 45, 50))

    draw = ImageDraw.Draw(src_img)
    for i, name in enumerate(agent_names):
        draw.text(
            (200*i + 55, 260),
            name,
            font=ImageFont.truetype("arial")
        )

    return src_img

def imgs2gif(imgs):
    imgs[0].save('out.gif', save_all=True, append_images=imgs[1:], duration=100, loop=0)
