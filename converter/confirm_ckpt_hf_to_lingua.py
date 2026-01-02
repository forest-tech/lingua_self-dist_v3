import torch
from torch import nn
from pathlib import Path
from omegaconf import OmegaConf
from apps.vanilla.transformer import LMTransformer, LMTransformerArgs
from lingua.checkpoint import CONSOLIDATE_NAME
from lingua.args import dataclass_from_dict
from transformers import AutoModelForCausalLM
            
            
lingua_consolidated_path = "llama_test_lingua"
hf_path = "llama_test_hf"


def load_consolidated_model_and_tokenizer(
    consolidated_path,
    model_cls=LMTransformer,
    model_args_cls=LMTransformerArgs,
):
    ckpt_path = Path(consolidated_path)
    config = ckpt_path / "params.json"
    config = OmegaConf.load(config)

    param_dtype = dict(fp32=torch.float32, fp16=torch.float16, bf16=torch.bfloat16)[
        config.distributed.model_dtype
    ]
    model_args = dataclass_from_dict(model_args_cls, config.model, strict=False)
    
    model = model_cls(model_args)
    st_dict = torch.load(ckpt_path / CONSOLIDATE_NAME, weights_only=True)
    model.load_state_dict(st_dict["model"])

    for param in model.parameters():
        param.data = param.data.to(dtype=param_dtype)
    return model, config

lingua_model, config = load_consolidated_model_and_tokenizer(consolidated_path=lingua_consolidated_path)

lingua_model.to("cuda:0") 
hf_model = AutoModelForCausalLM.from_pretrained(hf_path, torch_dtype=torch.bfloat16)
hf_model.to("cuda:0")
from transformers import AutoTokenizer

# tokenizer も用意
tokenizer = AutoTokenizer.from_pretrained(hf_path)

text = """No. 24; Updated March 2011
Click here to download and print a PDF version of this document.
Parents are usually the first to recognize that their child has a problem with emotions or behavior. Still, the decision to seek professional help can be difficult and painful for a parent. The first step is to gently try to talk to the child. An honest open talk about feelings can often help. Parents may choose to consult with the child's physicians, teachers, members of the clergy, or other adults who know the child well. These steps may resolve the problems for the child and family.
Following are a few signs which may indicate that a child and adolescent psychiatric evaluation will be useful.
- Marked fall in school performance
- Poor grades in school despite trying very hard
- Severe worry or anxiety, as shown by regular refusal to go to school, go to sleep or take part in activities that are normal for the child's age
- Frequent physical complaints
- Hyperactivity; fidgeting; constant movement beyond regular playing with or without difficulty paying attention
- Persistent nightmares
- Persistent disobedience or aggression (longer than 6 months) and provocative opposition to authority figures
- Frequent, unexplainable temper tantrums
- Threatens to harm or kill oneself
- Marked decline in school performance
- Inability to cope with problems and daily activities
- Marked changes in sleeping and/or eating habits
- Extreme difficulties in concentrating that get in the way at school or at home
- Sexual acting out
- Depression shown by sustained, prolonged negative mood and attitude, often accompanied by poor appetite, difficulty sleeping or thoughts of death
- Severe mood swings
- Strong worries or anxieties that get in the way of daily life, such as at school or socializing
- Repeated use of alcohol and/or drugs
- Intense fear of becoming obese with no relationship to actual body weight, excessive dieting, throwing up or using laxatives to loose weight
- Persistent nightmares
- Threats of self-harm or harm to others
- Self-injury or self destructive behavior
- Frequent outbursts of anger, aggression
- Repeated threats to run away
- Aggressive or non-aggressive consistent violation of rights of others; opposition to authority, truancy, thefts, or vandalism
- Strange thoughts, beliefs, feelings, or unusual behaviors
See other Facts for Families:
#25 Where to Seek Help for Your Child
#52 Comprehensive Psychiatric Evaluation
#57 Normal Adolescent Development, Middle School, and Early High School Years
#58 Normal Adolescent Development, Late High School Year and Beyond
#00 Definition of a Child and Adolescent Psychiatrist
The American Academy of Child and Adolescent Psychiatry (AACAP) represents over 8,500 child and adolescent psychiatrists who are physicians with at least five years of additional training beyond medical school in general (adult) and child and adolescent psychiatry.
Facts for Families© information sheets are developed, owned and distributed by AACAP. Hard copies of Facts sheets may be reproduced for personal or educational use without written permission, but cannot be included in material presented for sale or profit. All Facts can be viewed and printed from the AACAP website (www.aacap.org). Facts sheets may not be reproduced, duplicated or posted on any other website without written consent from AACAP. Organizations are permitted to create links to AACAP's website and specific Facts sheets. For all questions please contact the AACAP Communications & Marketing Coordinator, ext. 154.
If you need immediate assistance, please dial 911.
Copyright © 2012 by the American Academy of Child and Adolescent Psychiatry.
"""

input_ids = tokenizer.encode(text)



input_tensor = torch.tensor([input_ids])
input_tensor = input_tensor.to("cuda:0")
input_ids = input_tensor[..., :-1]
labels = input_tensor[..., 1:]


logits = hf_model(input_ids, labels=input_ids)
lingua_loss = lingua_model(input_ids, target=labels)
print(logits.loss)
print(lingua_loss)