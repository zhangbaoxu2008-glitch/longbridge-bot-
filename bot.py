import os
import json
import time
from datetime import datetime
from decimal import Decimal
from openai import OpenAI
from longbridge.openapi import (
    Config, QuoteContext, TradeContext,
    OrderType, OrderSide, TimeInForceType, Period, AdjustType
)

OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"].strip()
lb_config = Config.from_apikey_env()
quote_ctx = QuoteContext(lb_config)
trade_ctx = TradeContext(lb_config)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

MODEL_NAME = "deepseek/deepseek-chat"
RUN_MINUTES = int(os.environ.get("RUN_MINUTES", "18"))


def get_real_usd_cash() -> tuple[float, float]:
    """获取真实折算后的可用美元现金 (usd_cash) 与美元净资产 (usd_net_assets)"""
    usd_cash = 0.0
    usd_net = 0.0
    for b in trade_ctx.account_balance():
        rate = 7.82 if str(b.currency).upper() == "HKD" else 1.0
        usd_cash = float(b.total_cash) / rate
        usd_net = float(b.net_assets) / rate
        # 如果有细分币种现金列表，优先检查美元子账户与港币折算
        cash_infos = getattr(b, "cash_infos", [])
        if cash_infos:
            avail_usd_total = 0.0
            for ci in cash_infos:
                c_rate = 7.82 if str(ci.currency).upper() == "HKD" else 1.0
                avail_usd_total += max(0.0, float(ci.available_cash)) / c_rate
            if avail_usd_total > 0:
                usd_cash = avail_usd_total
    return round(usd_cash, 2), round(usd_net, 2)


def get_account_status() -> str:
    """获取已自动换算为美元(USD)的真实可用资金、当前持仓与今日订单"""
    usd_cash, usd_net = get_real_usd_cash()
    info = [
        f"【账户真实美元资产（已将港币按7.82折算为USD）】\n"
        f"当前真实可用美元现金: {usd_cash:.2f} USD (下单总金额绝对不能超过此数字！)\n"
        f"当前单笔买单建议上限(35%仓位): {usd_cash * 0.35:.2f} USD\n"
        f"账户折合美元总净资产: {usd_net:.2f} USD"
    ]

    pos_resp = trade_ctx.stock_positions()
    positions = []
    for ch in pos_resp.channels:
        for p in ch.positions:
            if int(p.quantity) > 0:
                positions.append(
                    f"代码: {p.symbol}, 持仓股数: {p.quantity}, 可卖股数: {p.available_quantity}, 成本均价: {p.cost_price} USD"
                )
    info.append("【当前持仓】\n" + ("\n".join(positions) if positions else "当前空仓"))

    orders = trade_ctx.today_orders()
    order_list = []
    for o in orders[:12]:
        order_list.append(
            f"订单号: {o.order_id}, 代码: {o.symbol}, 方向: {o.side}, "
            f"股数: {o.quantity}, 委托价: {o.price}, 状态: {o.status}"
        )
    info.append("【今日订单记录】\n" + ("\n".join(order_list) if order_list else "今日暂无订单"))
    return "\n\n".join(info)


def get_us_stock_quote(symbols: list[str]) -> str:
    quotes = quote_ctx.quote(symbols)
    lines = []
    for q in quotes:
        last_p = float(q.last_done)
        open_p = float(q.open) if float(q.open) > 0 else last_p
        prev_c = float(getattr(q, "prev_close", open_p)) or open_p
        day_chg = ((last_p - prev_c) / prev_c) * 100
        intra_chg = ((last_p - open_p) / open_p) * 100
        lines.append(
            f"代码: {q.symbol}, 最新价: {last_p:.2f} USD, 昨收涨跌: {day_chg:+.2f}%, "
            f"今开涨跌: {intra_chg:+.2f}%, 最高: {q.high}, 最低: {q.low}, 成交量: {q.volume}"
        )
    return "\n".join(lines)


def get_kline_trend(symbol: str) -> str:
    try:
        klines = quote_ctx.candlesticks(symbol, Period.Min_15, 12, AdjustType.NoAdjust)
    except Exception:
        klines = quote_ctx.candlesticks(symbol, Period.Min_15, 12)
    if not klines:
        return f"{symbol} 暂无K线数据"
    closes = [float(k.close) for k in klines]
    ma5 = sum(closes[-5:]) / min(len(closes), 5)
    ma10 = sum(closes[-10:]) / min(len(closes), 10)
    recent_bars = [
        f"收盘:{float(k.close):.2f}(高:{float(k.high):.2f}/低:{float(k.low):.2f})"
        for k in klines[-5:]
    ]
    return (
        f"【{symbol} 15分钟K线趋势】最新价: {closes[-1]:.2f} USD, MA5: {ma5:.2f}, MA10: {ma10:.2f}\n"
        f"最近5根K线: " + " -> ".join(recent_bars)
    )


def place_paper_limit_order(symbol: str, side: str, quantity: int, price: float, reason: str) -> str:
    """提交限价单，内置美元余额防超买自动校准器与订单状态实时复核"""
    if quantity <= 0:
        return "拦截：下单股数必须大于 0。"
    try:
        order_side = OrderSide.Buy if side.lower() == "buy" else OrderSide.Sell
        final_qty = int(quantity)

        # 如果是买入，自动检查真实可用美元，防止股数过大被长桥拒绝
        if order_side == OrderSide.Buy:
            usd_cash, _ = get_real_usd_cash()
            # 预留 5% 缓冲，且单笔最多动用当前可用美元的 45%
            max_safe_usd = usd_cash * 0.45
            max_affordable_qty = int(max_safe_usd // price)
            if final_qty > max_affordable_qty and max_affordable_qty > 0:
                print(f"⚠️ 自动校准股数: 原请求 {final_qty} 股超出单笔可用美元上限，已自动调整为 {max_affordable_qty} 股", flush=True)
                final_qty = max_affordable_qty

            # 再用长桥官方最大可买接口做二次校验
            try:
                est = trade_ctx.estimate_max_purchase_quantity(
                    symbol=symbol,
                    order_type=OrderType.LO,
                    side=OrderSide.Buy,
                    price=Decimal(str(round(price, 2)))
                )
                cash_max = int(getattr(est, "cash_max_qty", 0))
                if cash_max > 0 and final_qty > cash_max:
                    final_qty = int(cash_max * 0.95)
            except Exception:
                pass

        if final_qty <= 0:
            return "拦截：当前可用美元现金不足以买入该股票。"

        resp = trade_ctx.submit_order(
            symbol=symbol,
            order_type=OrderType.LO,
            side=order_side,
            submitted_quantity=Decimal(str(final_qty)),
            time_in_force=TimeInForceType.Day,
            submitted_price=Decimal(str(round(price, 2))),
            remark=f"FullAI: {reason[:28]}"
        )
        # 等待 1.5 秒复核长桥后台是否真正接受该订单
        time.sleep(1.5)
        for o in trade_ctx.today_orders():
            if o.order_id == resp.order_id:
                if "Reject" in str(o.status):
                    return f"下单被长桥拒绝！状态: {o.status}, 请将买入股数减半后重试！"
                return f"下单成功！[{side}] {symbol} {final_qty}股 @ {price:.2f} USD, 实时状态: {o.status}, 订单号: {resp.order_id}"

        return f"已提交订单！[{side}] {symbol} {final_qty}股 @ {price:.2f} USD, 订单号: {resp.order_id}"
    except Exception as e:
        return f"下单报错: {e}"


def cancel_pending_order(order_id: str) -> str:
    try:
        trade_ctx.cancel_order(order_id)
        return f"撤单成功！已撤销订单号: {order_id}"
    except Exception as e:
        return f"撤单失败: {e}"


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_account_status",
            "description": "查询已折算为真实美元(USD)的账户可用现金、总资产、当前持仓及今日订单记录",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_us_stock_quote",
            "description": "批量获取美股实时报价与涨跌幅",
            "parameters": {
                "type": "object",
                "properties": {"symbols": {"type": "array", "items": {"type": "string"}}},
                "required": ["symbols"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_kline_trend",
            "description": "获取指定美股的 15 分钟 K 线走势与 MA5/MA10 均线数据",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "place_paper_limit_order",
            "description": "提交美股限价单（Buy买入 / Sell卖出）",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "side": {"type": "string", "enum": ["Buy", "Sell"]},
                    "quantity": {"type": "integer"},
                    "price": {"type": "number"},
                    "reason": {"type": "string"}
                },
                "required": ["symbol", "side", "quantity", "price", "reason"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_pending_order",
            "description": "根据 order_id 撤销未成交的挂单",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"]
            }
        }
    }
]


def run_trading_cycle():
    print(f"\n================ 🚀 AI 全权基金经理巡检: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ================", flush=True)
    strategy_prompt = """
    你拥有我长桥美股模拟账户的 100% 最高全权托管权限！目标是实现账户净资产最大化。

    请严格按以下步骤操作：
    1. 调用 `get_account_status` 查看【当前真实可用美元现金】（已经自动从港币折算为真实 USD）。
       - 若有长时间未成交的挂单，调用 `cancel_pending_order` 撤销。
    2. 调用 `get_us_stock_quote` 扫描全市场高弹性标的池：
       - 核心科技与热门股：['NVDA.US', 'TSLA.US', 'AAPL.US', 'AMD.US', 'PLTR.US', 'META.US', 'MSFT.US', 'AVGO.US', 'COIN.US', 'MSTR.US']
       - 3倍杠杆多空ETF：['TQQQ.US', 'SQQQ.US', 'SOXL.US', 'SOXS.US']
    3. 对看好的 2~3 只标的或当前持仓调用 `get_kline_trend` 检查 15 分钟 K 线与均线趋势。
    4. 【仓位与下单规则】：
       - 持仓股若亏损达 -1.5% 或跌破均线走弱，果断卖出（Sell）换股；若盈利达 +2.5% 以上且动能放缓，果断止盈。
       - 买入新股时，请严格根据 `get_account_status` 返回的【当前真实可用美元现金】计算股数！单只股票分配 30%~40% 的可用美元现金（例如可用美元为 180,000 USD，单只分配 60,000 USD，除以股价算出整数股数）。
       - 为保证限价单立刻成交，买入价可设在比最新价高 0.10 USD，卖出价可设在比最新价低 0.10 USD。
    """
    messages = [{"role": "user", "content": strategy_prompt}]

    for _ in range(10):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=tools,
            temperature=0.2
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            print("\n🤖 AI 基金经理操盘报告：\n", msg.content, flush=True)
            break

        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments or "{}")
            print(f"⚙️ 执行指令: {fn_name}({args})", flush=True)

            if fn_name == "get_account_status":
                res = get_account_status()
            elif fn_name == "get_us_stock_quote":
                res = get_us_stock_quote(**args)
            elif fn_name == "get_kline_trend":
                res = get_kline_trend(**args)
            elif fn_name == "place_paper_limit_order":
                res = place_paper_limit_order(**args)
            elif fn_name == "cancel_pending_order":
                res = cancel_pending_order(**args)
            else:
                res = "未知工具"

            messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(res)})


start_time = time.time()
run_trading_cycle()

while (time.time() - start_time) < (RUN_MINUTES * 60):
    time.sleep(5)
    pos_resp = trade_ctx.stock_positions()
    cost_map = {
        p.symbol: float(p.cost_price)
        for ch in pos_resp.channels for p in ch.positions if int(p.quantity) > 0
    }
    if cost_map:
        quotes = quote_ctx.quote(list(cost_map.keys()))
        for q in quotes:
            last_p = float(q.last_done)
            cost_p = cost_map[q.symbol]
            pnl_pct = ((last_p - cost_p) / cost_p) * 100
            print(f"⚡ [5秒云端盯盘] {q.symbol} 现价:{last_p:.2f} USD, 成本:{cost_p:.2f} USD, 盈亏:{pnl_pct:+.2f}%", flush=True)
            if pnl_pct >= 2.5 or pnl_pct <= -1.5:
                print(f"🚨 异动触发 ({pnl_pct:+.2f}%)，立即唤醒 AI 基金经理决策！", flush=True)
                run_trading_cycle()
                break
