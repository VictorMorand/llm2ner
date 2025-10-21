import os, json, logging, spacy, time
from spacy import displacy
import torch
from datetime import datetime
import numpy as np
from transformer_lens import HookedTransformer

# display libs
import plotly.graph_objects as go
from IPython.display import display, HTML
import ipywidgets as widgets
from transformer_lens import HookedTransformer
import circuitsvis as cv
import matplotlib.pyplot as plt
import pandas as pd

from llm2ner import utils
from llm2ner.models import AttentionCNN_NER
from llm2ner.models.model import (
    NERmodel,
    count_perf,
    count_perf_tags,
)

logging.basicConfig(level=logging.INFO)


def display_entities(text, tokenizer, entities):
    """Uses Displacy to show Nested spans prediction
    Args:
        - text: original sentence
        - tokenizer: tokenizer from ToMMeR
        - enitites: predicted entities.
    """
    # Get token offsets for character-level mapping
    encoding = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
    offsets = encoding["offset_mapping"][0]  # shape: (num_tokens, 2)
    # Initialize spaCy
    nlp = spacy.blank("en")
    doc = nlp(text)
    char_spans = []
    # Add predicted entities
    for b, e in entities[0]:
        if b < len(offsets) and e < len(offsets):
            start_char = offsets[b][0].item()
            end_char = offsets[e][1].item()
            # Handle leading spaces
            if start_char < len(text) and text[start_char] == " ":
                start_char += 1
            char_spans.append((start_char, end_char, "PRED"))
    # Convert to spaCy spans
    doc.spans["sc"] = []
    for start, end, label in char_spans:
        span = doc.char_span(start, end, label=label, alignment_mode="expand")
        if span is None:
            print(f"⚠️ Could not align span ({start}, {end}): '{text[start:end]}'")
        else:
            doc.spans["sc"].append(span)
    # Display in notebook
    return displacy.render(doc, style="span", options={"colors": {"PRED": "lightblue",}})


def display_compare_entities(text, tokenizer, predicted_entities, gold_entities):
    """Uses Displacy to show both predicted and gold entity spans for comparison.
    
    Args:
        - text: original sentence
        - tokenizer: tokenizer from ToMMeR
        - predicted_entities: predicted entities (list of (start, end) tuples)
        - gold_entities: ground truth entities (list of dicts with 'pos' key containing (start_char, end_char))
    
    Returns:
        displacy HTML rendering
    """
    from llm2ner.results import count_perf
    
    # Get token offsets for character-level mapping
    encoding = tokenizer(text, return_offsets_mapping=True, return_tensors="pt")
    offsets = encoding["offset_mapping"][0]  # shape: (num_tokens, 2)
    
    # Initialize spaCy
    nlp = spacy.blank("en")
    doc = nlp(text)
    char_spans = []
    
    # Add gold entities
    for ent in gold_entities:
        if isinstance(ent, dict) and 'pos' in ent:
            start_char, end_char = ent["pos"]
        else:
            # Assume it's already a (start, end) tuple
            start_char, end_char = ent
        char_spans.append((start_char, end_char, "GOLD"))
    
    # Add predicted entities
    for b, e in predicted_entities[0] if isinstance(predicted_entities[0], list) else predicted_entities:
        if b < len(offsets) and e < len(offsets):
            start_char = offsets[b][0].item()
            end_char = offsets[e][1].item()
            # Handle leading spaces
            if start_char < len(text) and text[start_char] == " ":
                start_char += 1
            char_spans.append((start_char, end_char, "PRED"))
    
    # Convert to spaCy spans
    doc.spans["sc"] = []
    for start, end, label in char_spans:
        span = doc.char_span(start, end, label=label, alignment_mode="expand")
        if span is None:
            print(f"⚠️ Could not align span ({start}, {end}): '{text[start:end]}'")
        else:
            doc.spans["sc"].append(span)
    
    # Custom colors
    colors = {
        "PRED": "lightblue",
        "GOLD": "gold",
    }

    # Display in notebook
    return displacy.render(doc, style="span", options={"colors": colors}, jupyter=True)


def item_inference_html(model, item:dict, outputs:dict, save_html=True, output_dir="plots", ):
    """Fancy Display of outputs from test_inference"""
     
    text, str_tokens, gt_tags = (
        item["text"],
        item["str_tokens"],
        torch.tensor(item["token_tags"], dtype=torch.int32),
    )
    entities = outputs["entities"]
    tp, fp, tot = count_perf(entities, item["ent_pos"])
    print(f"- True pos: {tp} / {tot} \n- False pos: {fp} ")

    # === ENTITY VISUALIZATION ===
    # Use display_compare_entities for visualization
    html_content = display_compare_entities(text, model.tokenizer, entities, item["entities"])
    
    if save_html:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        item_id = item.get('id', 'unknown')
        filename = f"ner_viz_{item_id}_{timestamp}.html"
        filepath = os.path.join(output_dir, filename)
        
        # Enhanced HTML template for better printing
        enhanced_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>NER Visualization - Item {item_id}</title>
    <style>
        @media print {{
            body {{ margin: 2cm; }}
            .entities {{ page-break-inside: avoid; }}
        }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 40px;
            background-color: white;
        }}
        .header {{
            border-bottom: 2px solid #333;
            margin-bottom: 30px;
            padding-bottom: 20px;
        }}
        .info {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
            border-left: 4px solid #007bff;
        }}
        .entities {{
            font-size: 16px;
            line-height: 2.2;
            border: 1px solid #ddd;
            padding: 20px;
            border-radius: 5px;
            background-color: #fefefe;
        }}
        .legend {{
            margin-top: 20px;
            padding: 15px;
            background-color: #f1f3f4;
            border-radius: 5px;
        }}
        .legend-item {{
            display: inline-block;
            margin-right: 20px;
            margin-bottom: 10px;
        }}
        .legend-color {{
            display: inline-block;
            width: 20px;
            height: 15px;
            margin-right: 8px;
            vertical-align: middle;
            border: 1px solid #ccc;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Named Entity Recognition Visualization</h1>
        <p><strong>Item ID:</strong> {item_id} | <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="info">
        <h3>Performance Summary</h3>
        <p><strong>True Positives:</strong> {tp} / {tot} | <strong>False Positives:</strong> {fp}</p>
        <p><strong>Predicted Entities:</strong> {len(entities)} | <strong>Gold Entities:</strong> {len(item["entities"])}</p>
    </div>
    
    {html_content}
    
    <div class="legend">
        <h3>Legend</h3>
        <div class="legend-item">
            <span class="legend-color" style="background-color: lightblue;"></span>
            <strong>PRED</strong>: Predicted entities
        </div>
        <div class="legend-item">
            <span class="legend-color" style="background-color: gold;"></span>
            <strong>GOLD</strong>: Ground truth entities
        </div>
    </div>
</body>
</html>
        """
        
        # Save HTML file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(enhanced_html)
        
        print(f"\n📄 HTML visualization saved: {filepath}")
        
        return {"html_path": filepath}
    
    return outputs

############## interactive Inference ##############
@torch.no_grad()
def test_inference(
    ner_model: NERmodel,
    model: HookedTransformer,
    item: dict,
    decoding_strategy: str = "threshold",
    threshold: float = 0.5,
    normalize: bool = True,
    return_logits: bool = False,
    show_attn: bool = False,
    show_values: bool = False,
    verbose: bool = False,
):
    """Test inference on a single item
    Args:
        ner_model: TokenMatchingNER, model to test
        model: HookedTransformer, hooked model
        decoding_strategy: str, decoding strategy to use, see decoders.py for options
        threshold: float, threshold for the decoding strategy
        normalize: bool, whether to normalize the attention scores for visualization
        item: dict, item to test,
        display: how to display results: can be :
            - "cv" using circuits viz
            - "spacy": using displacy in "span" mode to show all predicted entities.
    Returns:
        tags: list, predicted tags
        scores: torch.tensor, attention scores
        end_ent: torch.tensor, end entity probe
    """
    text, str_tokens, gt_tags = (
        item["text"],
        item["str_tokens"],
        torch.tensor(item["token_tags"], dtype=torch.int32),
    )
    inputs = model.tokenizer(
        text, padding=True, padding_side="right", return_tensors="pt"
    )
    tokens = inputs["input_ids"].to(model.device)
    attn_mask = inputs["attention_mask"].to(model.device)

    seq = tokens.size(1)
    model_type = type(ner_model).__name__.split(".")[0]

    outputs = {}

    if model_type in ["ToMMeR", "CLQK_NER", "TokenMatchingNER"]:
        # inference

        Q, K, reps = ner_model.get_QK_reps(
            tokens, model, attn_mask=attn_mask
        )  # shape (batch, seq, rank) , " , (batch, seq, dim)
        
        end_ent = ner_model.classifier(reps).view(-1).detach()
        attn_scores, mask = ner_model.attn_forward(
            Q, K, return_mask=True, return_logits=False
        )  # shape (batch, seq, seq)
        attn_scores = attn_scores.view(seq, seq).detach()
        mask = mask.view(seq, seq).detach()

        entities = ner_model.infer_entities(
            tokens,
            model,
            decoding_strategy=decoding_strategy,
            threshold=threshold,
            attn_mask=attn_mask,
        )
        
        str_entities = [f"'{''.join(str_tokens[b:e+1])}' {(b,e)}" for b, e in entities[0]]
        if verbose:
            print(f"Found {len(entities[0])} entities: {', '.join(str_entities)}")

        tags = ner_model.get_tags(entities[0], seq).cpu()

        scores = (
            ner_model.forward(
                tokens, model, attn_mask=attn_mask, return_logits=return_logits
            )
            .view(seq, seq)
            .detach()
        )

        if return_logits and normalize:

            max_attn_scores = attn_scores[mask].abs().max()
            max_scores = scores[mask].max()
            if verbose:
                print(f"div by max abs attn scores: {max_attn_scores}")
                print(f"div by max scores: {max_scores}")
            n_attn_scores = attn_scores / max_attn_scores
            n_scores = scores / max_scores
        else:
            n_attn_scores = attn_scores
            n_scores = scores
        
       
        tp, fp, tot = count_perf(entities, item["ent_pos"])
        print(f"- True pos: {tp} / {tot} \n- False pos: {fp} ")

        # display results
        display_compare_entities(
            text, model.tokenizer, entities, item["entities"]
        )
        # Transform to RenderedHTML
        if show_values:
            display(
                cv.tokens.colored_tokens_multi(
                    tokens=str_tokens,
                    values=torch.vstack([ls for ls in (tags, gt_tags, end_ent.cpu())]).T,
                    labels=["Pred Tags", "ground Truth", "end probe"],
                )
            )

        if len(item["str_tokens"]) < 50 and show_attn:
            display(
            cv.attention.attention_heads(
                attention=torch.vstack(
                    (
                        n_scores.unsqueeze(0),
                        n_attn_scores.unsqueeze(0),
                        mask.unsqueeze(0),
                    )
                ),
                tokens=item["str_tokens"],
                attention_head_names=["span Scores", "Attn Scores", "mask"],
                mask_upper_tri=True,
                max_value=1,
                min_value=-1,
                )
            )
        else:
            if show_attn: print("Sequence too long to display attention scores")

        return outputs | {
            "tags":tags,
            "scores":scores,
            "end_ent":end_ent,
            "entities":entities,
        }

    elif model_type == "AttentionLCNER":
        # inference

        # get the attention scores from llms
        llm_attn_scores, reps = utils.get_attnScores_from_layers(
            layers=ner_model.layers,
            model=model,
            tokens=tokens,
            attn_mask=attn_mask,
        )  # shape (batch, len(layers), n_heads, seq, seq)

        attn_scores, mask = ner_model.attn_forward(llm_attn_scores, return_mask=True)  # shape (batch, seq, seq)
        end_ent = ner_model.classifier(reps)  # shape (batch, seq)

        entities = ner_model.infer_entities(
            tokens,
            model,
            decoding_strategy=decoding_strategy,
            threshold=threshold,
            attn_mask=attn_mask,
        )
        
        str_entities = [f"'{''.join(str_tokens[b:e+1])}' {(b,e)}" for b, e in entities[0]]
        print(f"Found {len(entities[0])} entities: {', '.join(str_entities)}")

        tags = ner_model.get_tags(entities[0], seq).cpu()

        scores = (
            ner_model.forward(
                tokens, model, attn_mask=attn_mask, return_logits=return_logits
            ).detach()
        )

        if return_logits and normalize:

            max_attn_scores = attn_scores[mask].abs().max()
            max_scores = scores[mask].max()
            print(f"div by max abs attn scores: {max_attn_scores}")
            print(f"div by max scores: {max_scores}")
            n_attn_scores = attn_scores / max_attn_scores
            n_scores = scores / max_scores
        else:
            n_attn_scores = attn_scores
            n_scores = scores
        # Transform to RenderedHTML

        display(
            cv.tokens.colored_tokens_multi(
                tokens=str_tokens,
                values=torch.vstack([ls for ls in (tags, gt_tags, end_ent.cpu())]).T,
                labels=["Pred Tags", "ground Truth", "end probe"],
            )
        )
        
        print(n_scores.shape, n_attn_scores.shape, mask.shape)
        tp, fp, tot = count_perf(entities, item["ent_pos"])
        print(f"- True pos: {tp} / {tot} \n- False pos: {fp} ")

        if len(item["str_tokens"]) < 50:
            display(
            cv.attention.attention_heads(
                attention=torch.vstack(
                    (
                        n_scores,
                        n_attn_scores,
                        mask,
                    )
                ),
                tokens=item["str_tokens"],
                attention_head_names=["span Scores", "Attn Scores", "mask"],
                mask_upper_tri=True,
                max_value=1,
                min_value=-1,
                )
            )
        else:
            print("Sequence too long to display attention scores")
        return tags, scores, end_ent

    elif model_type == "MHSA_NER":

        entities = ner_model.infer_entities(
            tokens,
            model,
            decoding_strategy=decoding_strategy,
            threshold=threshold,
            attn_mask=attn_mask,
        )
        tags = ner_model.get_tags(entities[0], seq).cpu()

        scores, mask = ner_model.forward(
            tokens,
            model,
            attn_mask=attn_mask,
            return_logits=return_logits,
            return_mask=True,
        )
        scores = scores.view(seq, seq).detach()
        mask = mask.view(seq, seq).detach()

        if not return_logits and normalize:
            max_scores = scores[mask].max()
            print(f"div by max scores: {max_scores}")
            n_scores = scores / max_scores
        else:
            n_scores = scores
        # Transform to RenderedHTML

        display(
            cv.tokens.colored_tokens_multi(
                tokens=str_tokens,
                values=torch.vstack([ls for ls in (tags, gt_tags)]).T,
                labels=["Pred Tags", "ground Truth"],
            )
        )
        tp, fp, tot = count_perf(entities, item["ent_pos"])
        print(f"- True pos: {tp} / {tot} \n- False pos: {fp} ")

        display(
            cv.attention.attention_heads(
                attention=torch.vstack((n_scores.unsqueeze(0), mask.unsqueeze(0))),
                tokens=item["str_tokens"],
                attention_head_names=["span Scores", "mask"],
                mask_upper_tri=False,
                max_value=1,
                min_value=-1,
            )
        )
        return outputs | {
            "tags":tags,
            "scores":scores,
            "mask":mask,
            "entities":entities,
        }

    elif model_type == "AttentionCNN_NER":
        entities = ner_model.infer_entities(tokens, model, attn_mask=attn_mask)
        tags = (
            ner_model.infer_tags(tokens, model, attn_mask=attn_mask)
            .view(-1)
            .detach()
            .cpu()
        )
        reps = ner_model.get_representations(tokens, model, attn_mask=attn_mask)
        end_ent = ner_model.classifier(reps).view(-1).detach()
        attn_scores, mask = ner_model.attn(reps, return_mask=True)
        attn_scores = attn_scores.detach()
        scores = ner_model.forward(
            tokens, model, attn_mask=attn_mask, return_logits=True
        ).detach()
        end_ent_probs = torch.sigmoid(end_ent)

        if normalize:
            # normalize attention scores
            logging.info(f"Normalizing logits for visualization")
            max_attn_scores = attn_scores[mask].abs().max()
            max_scores = scores[mask].max()
            print(f"div by max abs attn scores: {max_attn_scores}")
            print(f"div by max scores: {max_scores}")
            n_attn_scores = attn_scores / max_attn_scores
            n_scores = scores / max_scores
        else:
            n_attn_scores = attn_scores
            n_scores = scores

        # Transform to RenderedHTML
        display(
            cv.tokens.colored_tokens_multi(
                tokens=str_tokens,
                values=torch.vstack(
                    [ls for ls in (tags, gt_tags, end_ent_probs.cpu())]
                ).T,
                labels=["Pred Tags", "ground Truth", "end probe"],
            )
        )

        tp, fp, tot = count_perf(entities, item["ent_pos"])
        print(f"- True pos: {tp} / {tot} \n- False pos: {fp} ")

        display(
            cv.attention.attention_heads(
                attention=torch.vstack((n_scores, n_attn_scores, mask)),
                tokens=item["str_tokens"],
                attention_head_names=["scores", "Attn Scores", "Mask"],
                mask_upper_tri=True,
                max_value=1,
                min_value=-1,
            )
        )

        return outputs | {
            "tags":tags,
            "attn_scores":attn_scores,
            "end_ent":end_ent,
            "scores":scores,
            "mask":mask,
            "entities":entities,
        }

    elif model_type == "NERCmodel":
        entities, classes = ner_model.forward(
            tokens,
            model,
            attn_mask=attn_mask,
            decoding_strategy=decoding_strategy,
            threshold=threshold,
            return_logits=False,
        )
        classes = classes.view(-1).detach().cpu().numpy()
        gt_entities = item["entities"]
        gt_pos = [ent["tok_pos"] for ent in gt_entities]
        gt_classes = [ent["class"] for ent in gt_entities]

        tags = ner_model.infer_tags(tokens, model, attn_mask=attn_mask)
        assert (
            hasattr(ner_model, "class_reps") and ner_model.class_reps is not None
        ), "NERCmodel should have class_reps attribute set to a tensor of shape (n_classes, dim) "

        for i in range(len(entities)):
            _, b, e = entities[i]
            cls = classes[i]
            print(
                f"'{''.join(str_tokens[b:e+1])}' class {ner_model.id2type[cls]} ({cls})"
            )

        # Transform to RenderedHTML
        display(
            cv.tokens.colored_tokens_multi(
                tokens=str_tokens,
                values=torch.vstack([ls for ls in (tags, gt_tags)]).T,
                labels=["Pred Tags", "ground Truth"],
            )
        )

    else:
        logging.warning(f"unknown model type : {model_type}")
        return None, None, None

@torch.no_grad()
def demo_inference(
    text: str,
    ner_model: NERmodel,
    model: HookedTransformer,
    decoding_strategy: str = "threshold",
    threshold: float = 0.5,
    return_logits: bool = False,
    verbose: bool = False,
    show_attn: bool = False,
    show_values: bool = False,
):
    """Demo inference of ToMMer on a given string"""

    inputs = model.tokenizer(
        text, padding=True, padding_side="right", return_tensors="pt"
    )
    tokens = inputs["input_ids"].to(model.device)
    attn_mask = inputs["attention_mask"].to(model.device)
    str_tokens = utils.to_str_tokens(tokens, model.tokenizer)
    seq = tokens.size(1)
    model_type = type(ner_model).__name__.split(".")[0]

    if model_type in ["ToMMeR", "CLQK_NER", "TokenMatchingNER"]:
        # inference

        Q, K, reps = ner_model.get_QK_reps(
            tokens, model, attn_mask=attn_mask
        )  # shape (batch, seq, rank) , " , (batch, seq, dim)
        end_ent = ner_model.classifier(reps).view(-1).detach()
        attn_scores, mask = ner_model.attn_forward(
            Q, K, return_mask=True, return_logits=False
        )  # shape (batch, seq, seq)
        attn_scores = attn_scores.view(seq, seq).detach()
        mask = mask.view(seq, seq).detach()

        entities = ner_model.infer_entities(
            tokens,
            model,
            decoding_strategy=decoding_strategy,
            threshold=threshold,
            attn_mask=attn_mask,
        )
        tags = ner_model.get_tags(entities[0], seq).cpu()
        
        if verbose: 
            print(f"Found {len(entities[0])} entities: {entities[0]}")
            

        scores = (
            ner_model.forward(
                tokens, model, attn_mask=attn_mask, return_logits=return_logits
            )
            .view(seq, seq)
            .detach()
        )
        # First, show predicted spans.
        display(
            display_entities(text, model.tokenizer, entities)
        )
        if show_values:
            display(
                cv.tokens.colored_tokens_multi(
                    tokens=str_tokens,
                    values=torch.vstack([ls for ls in (tags, end_ent.cpu())]).T,
                    labels=["Pred Tags", "end probe"],
                )
            )
        if show_attn:
            display(
                cv.attention.attention_heads(
                    attention=torch.vstack(
                        (scores.unsqueeze(0), attn_scores.unsqueeze(0), mask.unsqueeze(0))
                    ),
                    tokens=str_tokens,
                    attention_head_names=["span Scores", "Attn Scores", "mask"],
                    mask_upper_tri=True,
                )
            )
        outputs = {
            "entities": entities,
            "tags": tags,
            "scores": scores.detach().cpu(),
            "end_ent": end_ent.detach().cpu(),
            "mask": mask.detach().cpu(),
            "attn_scores": attn_scores.detach().cpu(),
        }
        return outputs
    else:
        logging.warning("unknown model type:", model_type)
        return None


############### PLOTTING FUNCTIONS ####################


def plot_hist(hist, n_smooth=20, title=""):
    """Plot history of training"""

    if not len(hist):
        logging.warning("empty history")
        return

    # relpace 'val_loss' by 'val_metric' in hist if needed
    for h in hist:
        if "val_loss" in h:
            h["val_metric"] = h.pop("val_loss")

    keys = ["smooth_loss", "val_metric", "loss", "lr"]
    plots = {}
    for h in hist:
        for key in keys:
            if key in h:
                plots[key] = plots.get(key, [])
                plots[key].append((h["samples"], h[key]))

    # smooothing 'loss' and add in hist
    plots["smooth_loss"] = [
        (hist[i]["samples"], np.mean([h["loss"] for h in hist[i : i + n_smooth]]))
        for i in range(len(hist) - n_smooth)
    ]

    # loss_smooth = np.convolve(loss, np.ones(n_smooth)/n_smooth, mode='same')

    fig, ax1 = plt.subplots(figsize=(10, 5))
    color = "tab:blue"
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(*zip(*plots["smooth_loss"]), label="smooth_loss", color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    min_val, max_val = ax1.get_ylim()
    if min_val > 0:
        ax1.set_yscale("log")
    else:
        print("min loss <= 0, not using log scale")

    color = "tab:red"
    ax2 = ax1.twinx()
    ax2.set_ylabel("Validation Metric", color=color)
    ax2.plot(
        *zip(*plots["val_metric"]), label="val_metric", color=color, linestyle="--"
    )
    ax2.tick_params(axis="y", labelcolor=color)
    min_val, max_val = ax2.get_ylim()
    if max_val > 0:
        ax2.set_yscale("log")

    color = "tab:green"
    ax3 = ax1.twinx()
    ax3.spines["right"].set_position(("outward", 60))
    ax3.set_ylabel("lr", color=color)
    ax3.plot(*zip(*plots["lr"]), label="lr", color=color, linestyle="-.")
    ax3.tick_params(axis="y", labelcolor=color)
    ax3.set_yscale("log")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc="upper left")
    fig.tight_layout()
    plt.title(title)
    plt.show()


# Interactive plot with Plotly  # https://plotly.com/python/click-events/
def plot_all_interactive(
    metric: str,
    results_df: pd.DataFrame,
    x_key: str,
    y_key: str,
    color_by: str = "mode",
):
    layers = []
    hover_texts = []
    y = []
    colors = []
    model_name = results_df["model_name"].iloc[0]
    n_rows = len(results_df)
    if n_rows == 0:
        print("No results to plot")
        return

    hashs = []  # Store hashs for later use
    for ind, data in results_df.iterrows():
        if y_key not in data or data[y_key] is None:
            continue
        layers.append(data[x_key])
        y.append(data[y_key][metric])
        t = data["date"]

        date_hr = time.strftime("%b %d %Hh%M", time.localtime(t))
        metrics_to_display = data["Eval"].keys()
        metric_strings = [
            f"{metric}: {data['Eval'][metric]:.2f}"
            for metric in metrics_to_display
            if metric in data["Eval"]
        ]
        formatted_metrics = " | ".join(metric_strings)
        hover_text = (
            f"{data['mode']} | {date_hr} | dilation {data.get('dilate_entities','None')}<br>"
            f"x: {data[x_key]}| r: {data['rank']} | {'' if data['causal_mask'] else 'non'}causal | {data['dataset_name']}<br>"
            f"e: {data['epochs']} lr: {data['lr']} pos_weight {data['pos_weight']} b_size: {data['batch_size']}<br>"
            f"{formatted_metrics}<br>"
        )
        hover_texts.append(hover_text)
        path = data.get("path", "N/A")
        # Get last dir name
        hash = path.parts[-1] if path != "N/A" else "N/A"
        hashs.append(hash)  # Store the path info (default to "N/A" if not present)

        if color_by == "mode":
            if data["mode"] == "full":
                color = "blue"
            elif data["mode"] == "block":
                color = "green"
            elif data["mode"] == "block_only":
                color = "orange"
            else:
                color = "red"
        elif color_by == "causal":
            color = "blue" if data["causal_mask"] else "red"
        elif color_by == "pos_weight":
            if int(data["pos_weight"]) == 1:
                color = "green"
            elif int(data["pos_weight"]) == 2:
                color = "orange"
            elif int(data["pos_weight"]) == 3:
                color = "red"
        else:
            print(f"Unknown color_by value: {color_by}")
            color = "blue"
        colors.append(color)

    print(f"Plotting {len(y)} values for {model_name}...")
    # Create a scatter plot using FigureWidget for interactive events
    fig = go.FigureWidget(
        data=go.Scatter(
            # fig = go.Figure(data=go.Scatter(
            x=layers,
            y=y,
            mode="markers",
            marker=dict(
                size=7,
                color=colors,  # set color equal to a variable
            ),
            text=hover_texts,  # Hover text
            hoverinfo="text",
            customdata=hashs,  # Store path in customdata
        )
    )

    # add legend
    if color_by == "mode":
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color="blue"),
                name="full",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color="green"),
                name="block",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color="red"),
                name="block_only",
            )
        )
    elif color_by == "causal":
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color="red"),
                name="non causal",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color="blue"),
                name="causal",
            )
        )
    elif color_by == "pos_weight":
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color="green"),
                name="pos_weight 1",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color="orange"),
                name="pos_weight 2",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=10, color="red"),
                name="pos_weight 3",
            )
        )
    else:
        print("Unknown color_by value:", color_by)

    # Add titles and labels
    fig.update_layout(
        title=f"{metric} by layer for {model_name}",
        xaxis_title="Layer",
        yaxis_title=metric,
        legend_title="Legend",
        hovermode="closest",
        width=800,
        height=500,
    )

    # Create an output widget to display the hash
    output = widgets.Output()
    with output:
        print("Click on a point to display the corresponding hash.")

    # Define click event handler
    def on_click(trace, points, state):
        with output:
            output.clear_output()
            if points.point_inds:  # Check if any point was clicked
                ind = points.point_inds[0]  # Get the index of the clicked point
                hash = trace.customdata[
                    ind
                ]  # Get the corresponding path from customdata
                print(f"Point hash: {hash}")  # Print path to the output widget

    # Attach the click event handler
    fig.data[0].on_click(on_click)

    # Display the plot and the output widget in the Jupyter notebook
    display(fig)
    # fig.show()
    display(output)


def plot_patterns(ner_model: AttentionCNN_NER):
    cnn_pattern = ner_model.scoresKernel  # shape 1, 2 , k_width, k_height
    probe_pattern = ner_model.probeKernel  # shape 1, 2 , k_width, k_height
    left, right, top, bottom = ner_model.kernel_padding
    print(f"Aggregator pattern shape: {cnn_pattern.shape}")
    print(f"Probe pattern shape: {probe_pattern.shape}")

    plt.figure(
        figsize=(10, 11),
    )
    plt.subplot(2, 1, 1)
    plt.title("Scores Aggregator")
    max_abs = torch.max(torch.abs(cnn_pattern[0, 0]))
    plt.imshow(
        cnn_pattern[0, 0].detach().cpu().numpy(),
        vmin=-max_abs,
        vmax=max_abs,
        cmap="bwr",
    )
    plt.gca().add_patch(
        plt.Rectangle(
            (left - 0.5, top - 0.5), 1, 1, fill=False, edgecolor="lightgreen", lw=3
        )
    )
    plt.colorbar()

    plt.subplot(2, 1, 2)
    plt.title("end_ent probe pattern")
    plt.imshow(
        probe_pattern[0].detach().cpu().numpy(), vmin=-max_abs, vmax=max_abs, cmap="bwr"
    )
    plt.gca().add_patch(
        plt.Rectangle(
            (ner_model.probe_padding[1] - 0.5, -0.5),
            1,
            1,
            fill=False,
            edgecolor="lightgreen",
            lw=3,
        )
    )
    plt.tight_layout()

