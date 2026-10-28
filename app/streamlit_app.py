import json
import os
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Boréal Marché — Recommendation Engine",
    page_icon=":material/shopping_cart:",
    layout="wide",
)

# Resolve paths relative to project root
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

CATEGORY_EN_MAP = {
    "Viandes": "Meat",
    "Poissons": "Seafood",
    "Épicerie": "Pantry",
    "Boulangerie": "Bakery",
    "Fromages": "Cheese",
    "Collations": "Snacks",
    "Produits laitiers": "Dairy",
    "Fruits et légumes": "Produce",
    "Boissons": "Beverages",
    "Surgelés": "Frozen",
    "Bébé": "Baby",
    "Maison": "Household",
    "Spécialités": "Specialties",
}


@st.cache_data(ttl="1h", max_entries=5)
def load_catalogue_and_models():
    """Load product catalogue, offline-mined rules, and metadata from disk.
    Never calls Apriori online.
    """
    products_path = os.path.join(DATA_DIR, "products.csv")
    if not os.path.exists(products_path):
        products_path = os.path.join(PROJECT_ROOT, "products.csv")
    products_df = pd.read_csv(products_path)
    products_df["Category_EN"] = products_df["Category"].map(CATEGORY_EN_MAP).fillna(products_df["Category"])

    rules_path = os.path.join(MODELS_DIR, "rules.json")
    metadata_path = os.path.join(MODELS_DIR, "metadata.json")

    with open(rules_path, "r", encoding="utf-8") as f:
        rules_list = json.load(f)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return products_df, rules_list, metadata


def get_strength_badge(lift_val: float):
    """Return a human-readable association strength label and badge color derived from lift."""
    if lift_val >= 4.0:
        return "Very strong association", "green"
    elif lift_val >= 2.5:
        return "Strong association", "blue"
    elif lift_val >= 1.5:
        return "Moderate association", "orange"
    else:
        return "Weak association", "gray"


def match_rules_for_basket(basket_items: list[str], rules_list: list[dict]):
    """Subset matching engine:
    1. Fires every rule whose antecedents are a subset of the current basket.
    2. Filters out any rule whose consequent is already in the basket (Edge Case 3).
    3. Deduplicates multiple rules suggesting the same item by keeping the highest lift (Edge Case 4).
    """
    basket_set = set(basket_items)
    if not basket_set:
        return [], [], 0

    all_matching_rules = []
    for rule in rules_list:
        ant_set = set(rule["antecedents"])
        consequent = rule["consequent"]
        if ant_set.issubset(basket_set) and consequent not in basket_set:
            all_matching_rules.append(rule)

    # Sort all matching rules by descending lift, then confidence, then support
    all_matching_rules.sort(
        key=lambda r: (r["lift"], r["confidence"], r["support"]),
        reverse=True,
    )

    # Deduplicate consequent items: keep the rule with the highest lift for each suggested item
    deduped_recommendations = []
    seen_consequents = set()
    duplicates_collapsed = 0

    for rule in all_matching_rules:
        cons = rule["consequent"]
        if cons not in seen_consequents:
            seen_consequents.add(cons)
            deduped_recommendations.append(rule)
        else:
            duplicates_collapsed += 1

    return deduped_recommendations[:5], all_matching_rules, duplicates_collapsed


# Initialize session state for basket
st.session_state.setdefault("basket", [])
st.session_state.setdefault("preset_applied", None)

products_df, rules_list, metadata = load_catalogue_and_models()

# Build fast lookup maps
product_lookup = products_df.set_index("Item").to_dict(orient="index")
freq_lookup = {
    entry["item"]: entry
    for entry in metadata.get("all_product_frequencies", [])
}
antecedent_universe = set(metadata.get("coverage", {}).get("recommendable_products", [])) | {
    item for r in rules_list for item in r["antecedents"]
}
min_support_threshold = metadata.get("thresholds", {}).get("min_support", 0.01)

# Support URL query parameter presets for instant testing and automated screenshots
query_preset = st.query_params.get("preset", None)
if query_preset and st.session_state.preset_applied != query_preset:
    if query_preset == "empty":
        st.session_state.basket = []
    elif query_preset == "charcoal":
        st.session_state.basket = ["Charbon de bois"]
    elif query_preset == "bbq3":
        st.session_state.basket = ["Charbon de bois", "Steak de boeuf", "Pain à hamburger"]
    elif query_preset == "rare":
        st.session_state.basket = ["Caviar"]
    elif query_preset == "norule":
        st.session_state.basket = ["Carottes"]
    elif query_preset == "dedup":
        st.session_state.basket = ["Fromage à raclette", "Bouillon à fondue", "Charcuteries"]
    st.session_state.preset_applied = query_preset


def add_to_basket(item_fr: str):
    if item_fr not in st.session_state.basket:
        st.session_state.basket.append(item_fr)


def remove_from_basket(item_fr: str):
    if item_fr in st.session_state.basket:
        st.session_state.basket.remove(item_fr)


def clear_basket():
    st.session_state.basket = []


def set_demo_basket(items: list[str]):
    st.session_state.basket = list(items)


# --- HEADER SECTION ---
st.title("Boréal Marché — Recommendation Engine")
st.caption(
    "Association rules · Apriori algorithm · "
    f"{metadata['dataset_metrics']['total_baskets']:,} baskets · "
    f"{metadata['dataset_metrics']['total_products']} products · "
    "rules are mined offline and loaded from a file"
)

basket_items = st.session_state.basket
basket_total = sum(product_lookup[item]["Price"] for item in basket_items if item in product_lookup)

kpi_cols = st.columns(4)
with kpi_cols[0].container(border=True):
    st.metric("Rules loaded", f"{len(rules_list):,}")
with kpi_cols[1].container(border=True):
    st.metric("Products", f"{len(products_df)}")
with kpi_cols[2].container(border=True):
    st.metric("Items in basket", f"{len(basket_items)}")
with kpi_cols[3].container(border=True):
    st.metric("Basket total", f"${basket_total:.2f}")

# Quick scenario testing bar for grading & live client demonstration
with st.container(horizontal=True, vertical_alignment="center"):
    st.caption("**Quick test scenarios:**")
    st.button(
        "Fig 1: Empty basket (Cold start)",
        icon=":material/shopping_basket:",
        on_click=set_demo_basket,
        args=([],),
    )
    st.button(
        "Fig 2: Charcoal (1 item)",
        icon=":material/outdoor_grill:",
        on_click=set_demo_basket,
        args=(["Charbon de bois"],),
    )
    st.button(
        "Fig 3 & 4: BBQ basket (3 items)",
        icon=":material/lunch_dining:",
        on_click=set_demo_basket,
        args=(["Charbon de bois", "Steak de boeuf", "Pain à hamburger"],),
    )
    st.button(
        "Edge case: Rare item (Caviar, 0.08%)",
        icon=":material/diamond:",
        on_click=set_demo_basket,
        args=(["Caviar"],),
    )
    st.button(
        "Edge case: No rule fires (Carrots)",
        icon=":material/nutrition:",
        on_click=set_demo_basket,
        args=(["Carottes"],),
    )

st.space("small")

# --- THREE REQUIRED ZONES ---
zone1, zone2, zone3 = st.columns([1.15, 1.0, 1.35], gap="medium")

# ==============================================================================
# ZONE 1 · CATALOGUE
# ==============================================================================
with zone1:
    st.subheader("1 · Catalogue")

    categories_en = ["All categories"] + sorted(products_df["Category_EN"].unique().tolist())
    selected_category = st.selectbox(
        "Filter by category",
        options=categories_en,
        label_visibility="collapsed",
        key="catalogue_category",
    )

    search_query = st.text_input(
        "Search products by name",
        placeholder="Search by English or French name...",
        label_visibility="collapsed",
        key="catalogue_search",
    )

    filtered_catalogue = products_df.copy()
    if selected_category != "All categories":
        filtered_catalogue = filtered_catalogue[filtered_catalogue["Category_EN"] == selected_category]

    if search_query.strip():
        q = search_query.strip().lower()
        filtered_catalogue = filtered_catalogue[
            filtered_catalogue["Item_EN"].str.lower().str.contains(q, na=False)
            | filtered_catalogue["Item"].str.lower().str.contains(q, na=False)
            | filtered_catalogue["Category_EN"].str.lower().str.contains(q, na=False)
        ]

    st.caption(f"{len(filtered_catalogue)} product(s)")

    with st.container(height=540, border=False):
        for _, prod in filtered_catalogue.iterrows():
            item_fr = prod["Item"]
            item_en = prod["Item_EN"]
            cat_en = prod["Category_EN"]
            price = prod["Price"]
            in_basket = item_fr in basket_items

            with st.container(border=True):
                row_cols = st.columns([3.2, 1.3], vertical_alignment="center")
                with row_cols[0]:
                    st.markdown(f"**{item_en}**  \n:gray[{cat_en} · *{item_fr}* · **${price:.2f}**]")
                with row_cols[1]:
                    if in_basket:
                        st.button(
                            "Added",
                            key=f"cat_add_{item_fr}",
                            disabled=True,
                            icon=":material/check:",
                            width="stretch",
                        )
                    else:
                        st.button(
                            "Add",
                            key=f"cat_add_{item_fr}",
                            on_click=add_to_basket,
                            args=(item_fr,),
                            icon=":material/add:",
                            width="stretch",
                        )

# ==============================================================================
# ZONE 2 · MY BASKET
# ==============================================================================
with zone2:
    st.subheader("2 · My basket")

    if not basket_items:
        with st.container(border=True):
            st.caption("Your basket is empty. Add a product to see recommendations.")

        st.markdown("##### Most popular products")
        for pop in metadata.get("popular_products", [])[:5]:
            with st.container(border=True):
                p_cols = st.columns([3.2, 1.3], vertical_alignment="center")
                with p_cols[0]:
                    st.markdown(
                        f"**{pop['item_en']}** (:gray[*{pop['item']}*])  \n"
                        f":gray[{pop['support_pct']:.1f}% of baskets · ${pop['price']:.2f}]"
                    )
                with p_cols[1]:
                    st.button(
                        "Add",
                        key=f"pop_basket_{pop['item']}",
                        on_click=add_to_basket,
                        args=(pop["item"],),
                        icon=":material/add:",
                        width="stretch",
                    )
    else:
        for item_fr in basket_items:
            info = product_lookup[item_fr]
            with st.container(border=True):
                b_cols = st.columns([3.6, 1.0], vertical_alignment="center")
                with b_cols[0]:
                    st.markdown(
                        f"**{info['Item_EN']}**  \n"
                        f":gray[{info['Category_EN']} · *{item_fr}* · **${info['Price']:.2f}**]"
                    )
                with b_cols[1]:
                    st.button(
                        "✕",
                        key=f"rm_{item_fr}",
                        on_click=remove_from_basket,
                        args=(item_fr,),
                        help=f"Remove {info['Item_EN']} from basket",
                        width="stretch",
                    )

        with st.container(border=True):
            tot_cols = st.columns([2, 1], vertical_alignment="center")
            tot_cols[0].markdown(f"**Total ({len(basket_items)} item{'s' if len(basket_items) != 1 else ''})**")
            tot_cols[1].markdown(f"**${basket_total:.2f}**")

        st.button(
            "Clear basket",
            icon=":material/delete_sweep:",
            on_click=clear_basket,
            width="stretch",
        )

# ==============================================================================
# ZONE 3 · RECOMMENDATIONS ("YOU MAY ALSO LIKE")
# ==============================================================================
with zone3:
    st.subheader("3 · You may also like")

    # Edge Case 1: Empty basket ("Cold start")
    if not basket_items:
        st.info(
            "No rule can fire on an empty basket. This is the **'cold start'** edge case: "
            "the system falls back to the most popular products.",
            icon=":material/info:",
        )
        st.markdown("##### Popular fallback suggestions")
        for pop in metadata.get("popular_products", [])[:5]:
            with st.container(border=True):
                f_cols = st.columns([3.5, 1.2], vertical_alignment="center")
                with f_cols[0]:
                    cat_en = CATEGORY_EN_MAP.get(pop["category"], pop["category"])
                    st.markdown(
                        f"**{pop['item_en']}** · **${pop['price']:.2f}**  \n"
                        f":gray[Category: {cat_en} (*{pop['item']}*)]  \n"
                        f"Purchased in **{pop['support_pct']:.1f}%** of all Boréal Marché baskets "
                        f"({pop['basket_count']:,} orders)."
                    )
                with f_cols[1]:
                    st.button(
                        "Add",
                        key=f"cold_rec_{pop['item']}",
                        on_click=add_to_basket,
                        args=(pop["item"],),
                        icon=":material/add_shopping_cart:",
                        width="stretch",
                    )
    else:
        top_recs, all_fired_rules, num_deduped = match_rules_for_basket(basket_items, rules_list)

        # Check if basket contains any ultra-rare items (< min_support)
        rare_in_basket = [
            item for item in basket_items
            if freq_lookup.get(item, {}).get("support", 0) < min_support_threshold
        ]

        if rare_in_basket:
            for r_item in rare_in_basket:
                r_info = freq_lookup[r_item]
                st.warning(
                    f"**Why '{r_info['item_en']}' (*{r_item}*) appears in no rule at all (Rare Product Edge Case):**  \n"
                    f"'{r_info['item_en']}' appears in only **{r_info['basket_count']:,} out of "
                    f"{metadata['dataset_metrics']['total_baskets']:,} baskets ({r_info['support']*100:.2f}% support)**. "
                    f"Because Apriori was mined with `min_support = {min_support_threshold*100:.1f}%` "
                    f"({int(min_support_threshold * metadata['dataset_metrics']['total_baskets']):,} baskets), "
                    "this rare specialty product was pruned in Pass 1 and cannot trigger any association rule.",
                    icon=":material/warning:",
                )

        # Edge Case 2: Non-empty basket, but no rule fires
        if not top_recs:
            if not rare_in_basket:
                basket_names_en = ", ".join(product_lookup[i]["Item_EN"] for i in basket_items)
                st.warning(
                    f"**No association rule fired for your current basket ({basket_names_en}).**  \n"
                    "Although these products pass the 1.0% minimum support threshold individually, no consequent "
                    "reaches our `25%` minimum confidence & `lift > 1.05` thresholds from these items alone. "
                    "Falling back to the store's most popular products:",
                    icon=":material/search_off:",
                )
            else:
                st.info(
                    "Because no association rule could fire from the current basket, the system falls back "
                    "to the store's most popular products (excluding items already in your basket):",
                    icon=":material/lightbulb:",
                )

            shown_fallback = 0
            for pop in metadata.get("popular_products", []):
                if pop["item"] in basket_items:
                    continue
                with st.container(border=True):
                    f_cols = st.columns([3.5, 1.2], vertical_alignment="center")
                    with f_cols[0]:
                        cat_en = CATEGORY_EN_MAP.get(pop["category"], pop["category"])
                        st.markdown(
                            f"**{pop['item_en']}** · **${pop['price']:.2f}**  \n"
                            f":gray[Category: {cat_en} (*{pop['item']}*)]  \n"
                            f"Fallback suggestion — appears in **{pop['support_pct']:.1f}%** of all baskets."
                        )
                    with f_cols[1]:
                        st.button(
                            "Add",
                            key=f"fallback_rec_{pop['item']}",
                            on_click=add_to_basket,
                            args=(pop["item"],),
                            icon=":material/add_shopping_cart:",
                            width="stretch",
                        )
                shown_fallback += 1
                if shown_fallback >= 5:
                    break
        else:
            if num_deduped > 0:
                st.caption(
                    f":material/filter_alt: *{len(all_fired_rules)} rules matched your basket; "
                    f"{num_deduped} duplicate item suggestion(s) were deduplicated by keeping the rule with the highest lift.*"
                )

            # Render up to 5 recommendations sorted by descending lift
            for rec in top_recs:
                cons_fr = rec["consequent"]
                cons_info = product_lookup[cons_fr]
                cons_en = cons_info["Item_EN"]
                cat_en = cons_info["Category_EN"]
                price = cons_info["Price"]

                ant_names_en = ", ".join(rec["antecedents_en"])
                conf_pct = rec["confidence"] * 100.0
                supp_pct = rec["support"] * 100.0
                lift_val = rec["lift"]
                strength_label, badge_color = get_strength_badge(lift_val)

                with st.container(border=True):
                    head_cols = st.columns([3.5, 1.2], vertical_alignment="center")
                    with head_cols[0]:
                        st.markdown(
                            f"#### {cons_en} · **${price:.2f}**  \n"
                            f":gray[{cat_en} · *{cons_fr}*]"
                        )
                    with head_cols[1]:
                        st.button(
                            "Add",
                            key=f"rec_add_{cons_fr}",
                            on_click=add_to_basket,
                            args=(cons_fr,),
                            icon=":material/add_shopping_cart:",
                            type="primary",
                            width="stretch",
                        )

                    # Plain-language comment required by Section 5.2
                    st.markdown(
                        f"**Why this?** Because your basket contains **{ant_names_en}**, "
                        f"customers who buy that also buy **{cons_en}** **{conf_pct:.0f}%** of the time — "
                        f"that is **{lift_val:.1f}×** more often than average."
                    )

                    # Three metrics + strength label derived from lift
                    st.markdown(
                        f":blue-badge[Lift {lift_val:.2f}] "
                        f":violet-badge[Confidence {conf_pct:.0f}%] "
                        f":gray-badge[Support {supp_pct:.2f}%] "
                        f":{badge_color}-badge[{strength_label}]"
                    )

            # Expandable "Rules that fired" audit table (Required by Section 5.2 Figure 4 & Law 25)
            with st.expander(f"Rules that fired ({len(all_fired_rules)} matching rules)", expanded=True, icon=":material/manage_search:"):
                st.caption(
                    "Law 25 (Québec) Automated Decision Transparency: exact offline association rules "
                    "that triggered the recommendations above, sorted by descending lift."
                )
                fired_rows = []
                for r in all_fired_rules[:25]:
                    fired_rows.append({
                        "If the basket contains": " + ".join(r["antecedents_en"]),
                        "Then suggest": r["consequent_en"],
                        "Support": f"{r['support']*100:.2f}%",
                        "Confidence": f"{r['confidence']*100:.1f}%",
                        "Lift": round(r["lift"], 2),
                    })
                st.dataframe(pd.DataFrame(fired_rows), hide_index=True, width="stretch")
