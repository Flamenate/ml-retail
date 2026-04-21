from flask import Flask, jsonify, render_template, request

from .services import (
    batch_summary,
    load_customer_by_id,
    load_model_summary,
    load_processed_data,
    load_segment_overview,
    normalize_record,
    predict_batch,
    predict_customer_record,
)

PAGE_SIZE = 10
PROCESS_ALL_LIMIT = 100
RISK_PRIORITY = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "retail-churn-dashboard"

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            model_summary=load_model_summary(),
            segments=load_segment_overview(),
            result=None,
            error=None,
            batch_results=None,
            batch_summary=None,
            batch_pagination=None,
            customer_id="",
        )

    @app.post("/predict")
    def predict():
        customer_id = request.form.get("customer_id", "").strip()
        action = request.form.get("action", "single")

        try:
            if action == "process_all":
                try:
                    page = max(int(request.form.get("page", 1)), 1)
                except (TypeError, ValueError):
                    page = 1
                frame = load_processed_data().head(PROCESS_ALL_LIMIT).copy()
                predictions = predict_batch(frame)
                predictions = predictions.assign(
                    _risk_order=predictions["RiskLevel"].astype(str).map(RISK_PRIORITY).fillna(99)
                ).sort_values(
                    by=["_risk_order", "ChurnProbability", "CustomerID"],
                    ascending=[True, False, True],
                )
                summary = batch_summary(predictions)
                total_rows = len(predictions)
                total_pages = max((total_rows + PAGE_SIZE - 1) // PAGE_SIZE, 1)
                if page > total_pages:
                    page = total_pages
                start_idx = (page - 1) * PAGE_SIZE
                end_idx = start_idx + PAGE_SIZE
                preview = predictions.iloc[start_idx:end_idx].drop(columns="_risk_order").to_dict(orient="records")
                return render_template(
                    "index.html",
                    model_summary=load_model_summary(),
                    segments=load_segment_overview(),
                    result=None,
                    error=None,
                    batch_results=[normalize_record(row) for row in preview],
                    batch_summary=summary,
                    batch_pagination={
                        "page": page,
                        "page_size": PAGE_SIZE,
                        "total_pages": total_pages,
                        "total_rows": total_rows,
                        "start_row": start_idx + 1 if total_rows else 0,
                        "end_row": min(end_idx, total_rows),
                        "has_prev": page > 1,
                        "has_next": page < total_pages,
                        "prev_page": page - 1,
                        "next_page": page + 1,
                    },
                    customer_id=customer_id,
                )

            if not customer_id:
                raise ValueError("Provide a Customer ID or click Process All.")

            record = load_customer_by_id(int(customer_id))
            result = predict_customer_record(record)
            profile = result.pop("CustomerProfile", {})
            return render_template(
                "index.html",
                model_summary=load_model_summary(),
                segments=load_segment_overview(),
                result=result,
                customer_profile=profile,
                error=None,
                batch_results=None,
                batch_summary=None,
                batch_pagination=None,
                customer_id=customer_id,
            )
        except (ValueError, FileNotFoundError, KeyError) as exc:
            return render_template(
                "index.html",
                model_summary=load_model_summary(),
                segments=load_segment_overview(),
                result=None,
                customer_profile=None,
                error=str(exc),
                batch_results=None,
                batch_summary=None,
                batch_pagination=None,
                customer_id=customer_id,
            )

    @app.post("/api/predict")
    def api_predict():
        payload = request.get_json(silent=True)
        if not payload:
            return jsonify({"error": "A JSON body is required."}), 400

        if "customer_id" in payload:
            record = load_customer_by_id(int(payload["customer_id"]))
        elif "record" in payload:
            record = payload["record"]
            if not isinstance(record, dict):
                return jsonify({"error": "`record` must be a JSON object."}), 400
        else:
            return jsonify({"error": "Provide either `customer_id` or `record`."}), 400

        result = predict_customer_record(record)
        return jsonify(result)

    @app.get("/api/summary")
    def api_summary():
        return jsonify(
            {
                "model": load_model_summary(),
                "segments": load_segment_overview(),
                "customers": len(load_processed_data()),
            }
        )

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"})

    return app


app = create_app()

