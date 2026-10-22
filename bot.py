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
RUN_MINUTES = int(os.environ.get("RUN_MINUTES", "58"))

# 默认高弹性观察池（AI 随时可突破此列表查询全市场任意美股）
DEFAULT_WATCHLIST = [
    "NVDA.US", "TSLA.US", "AMD.US", "PLTR.US", "META.US", "AVGO.US",
    "COIN.US", "MSTR.US", "SMCI.US", "APP.US", "TQQQ.US", "SQQQ.US",
    "SOXL.US", "SOXS.US", "TSLL.US", "NVDL.US"
]


def get_real_usd_cash() -> tuple[float, float]:
    """获取真实折算后的可用美元现金（含融资购买力）与美元净资产"""
    usd_cash = 0.0
    usd_net = 0.0
    for b in trade_ctx.account_balance():
        rate = 7.82 if str(b.currency).upper() == "HKD" else 1.0
        usd_cash = float(b.total_cash) / rate
        usd_net = float(b.net_assets) / rate

        cash_infos = getattr(b, "cash_infos", [])
        if cash_infos:
            net_avail_usd = 0.0
            for ci in cash_infos:
                c_rate = 7.82 if str(ci.currency).upper() == "HKD" else 1.0
                net_avail_usd += float(ci.available_cash) / c_rate
            usd_cash = min(usd_cash, net_avail_usd)

        rem_finance = getattr(b, "remaining_finance_amount", None)
        if rem_finance is not None:
            usd_cash = min(usd_cash, float(rem_finance) / rate)

    return round(max(0.0, usd_cash), 2), round(usd_net, 2)


def get_account_status() -> str:
    """获取真实可用资金、当前持仓（含实时浮盈）与今日有效订单"""
    usd_cash, usd_net = get_real_usd_cash()
    info = [
        f"【账户真实美元资产】\n"
        f"当前最高可用美元购买力: {usd_cash:.2f} USD\n"
        f"账户折合美元总净资产: {usd_net:.2f} USD"
    ]

    pos_resp = trade_ctx.stock_positions()
    positions = []
    pos_symbols = [p.symbol for ch in pos_resp.channels for p in ch.positions if int(p.quantity) > 0]
    quote_map = {q.symbol: float(q.last_done) for q in quote_ctx.quote(pos_symbols)} if pos_symbols else {}

    for ch in pos_resp.channels:
        for p in ch.positions:
            qty = int(p.quantity)
            if qty > 0:
                cost = float(p.cost_price)
                cur_price = quote_map.get(p.symbol, cost)
                pnl_pct = ((cur_price - cost) / cost * 100) if cost > 0 else 0.0
                positions.append(
                    f"代码: {p.symbol}, 持仓股数: {qty}, 可卖股数: {p.available_quantity}, "
                    f"成本均价: {cost:.2f} USD, 现价: {cur_price:.2f} USD, 浮盈亏: {pnl_pct:+.2f}%"
                )
    info.append("【当前持仓】\n" + ("\n".join(positions) if positions else "当前空仓"))

    orders = trade_ctx.today_orders()
    order_list = []
    for o in orders:
        if "Reject" not in str(o.status) and "Cancel" not in str(o.status):
            order_list.append(
                f"订单号: {o.order_id}, 代码: {o.symbol}, 方向: {o.side}, "
                f"股数: {o.quantity}, 委托价: {o.price}, 状态: {o.status}"
            )
    info.append("【今日有效及未成交订单记录】\n" + ("\n".join(order_list) if order_list else "当前无未成交挂单"))
    return "\n\n".join(info)


def get_us_stock_quote(symbols: list[str]) -> str:
    if not symbols:
        return "未提供股票代码"
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


def calc_rsi(closes: list[float], period: int = 14) -> float:
    if len(closes) <= period:
        return 50.0
    gains, losses = [], []
    for i in range(-period, 0):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(max(-diff, 0.0))
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - (100.0 / (1.0 + rs)), 1)


def get_kline_trend(symbol: str, period: str = "15m") -> str:
    period_map = {
        "5m": Period.Min_5,
        "15m": Period.Min_15,
        "60m": Period.Min_60,
        "day": Period.Day
    }
    p_enum = period_map.get(period, Period.Min_15)
    try:
        klines = quote_ctx.candlesticks(symbol, p_enum, 30, AdjustType.NoAdjust)
    except Exception:
        klines = quote_ctx.candlesticks(symbol, p_enum, 30)
    if not klines:
        return f"{symbol} 暂无K线数据"
    closes = [float(k.close) for k in klines]
    highs = [float(k.high) for k in klines]
    lows = [float(k.low) for k in klines]
    ma5 = sum(closes[-5:]) / min(len(closes), 5)
    ma10 = sum(closes[-10:]) / min(len(closes), 10)
    ma20 = sum(closes[-20:]) / min(len(closes), 20)
    rsi14 = calc_rsi(closes, 14)
    recent_bars = [
        f"收:{float(k.close):.2f}(高:{float(k.high):.2f}/低:{float(k.low):.2f}/量:{k.volume})"
        for k in klines[-8:]
    ]
    return (
        f"【{symbol} ({period}) 量化趋势】最新价: {closes[-1]:.2f} USD | "
        f"MA5: {ma5:.2f}, MA10: {ma10:.2f}, MA20: {ma20:.2f} | RSI(14): {rsi14} | "
        f"近20根最高: {max(highs[-20:]):.2f}, 最低: {min(lows[-20:]):.2f}\n"
        f"最近8根K线: " + " -> ".join(recent_bars)
    )


def place_paper_limit_order(symbol: str, side: str, quantity: int, price: float, reason: str) -> str:
    """提交限价单，无资金底线与购买力校准限制"""
    if quantity <= 0:
        return "拦截：下单股数必须大于 0。"
    try:
        order_side = OrderSide.Buy if side.lower() == "buy" else OrderSide.Sell
        final_qty = int(quantity)

        resp = trade_ctx.submit_order(
            symbol=symbol,
            order_type=OrderType.LO,
            side=order_side,
            submitted_quantity=Decimal(str(final_qty)),
            time_in_force=TimeInForceType.Day,
            submitted_price=Decimal(str(round(price, 2))),
            remark=f"MaxAlpha: {reason[:25]}"
        )
        time.sleep(1.5)
        for o in trade_ctx.today_orders():
            if o.order_id == resp.order_id:
                if "Reject" in str(o.status):
                    return f"下单被长桥拒绝（状态: {o.status}），请勿重复提交该笔订单！"
                return f"下单成功！[{side}] {symbol} {final_qty}股 @ {price:.2f} USD, 状态: {o.status}, 订单号: {resp.order_id}"

        return f"已提交订单！[{side}] {symbol} {final_qty}股 @ {price:.2f} USD, 订单号: {resp.order_id}"
    except Exception as e:
        return f"下单报错: {e}"


def cancel_pending_order(order_id: str) -> str:
    try:
        trade_ctx.cancel_order(order_id)
        time.sleep(1.5)
        return f"撤单请求已提交！已撤销订单号: {order_id}"
    except Exception as e:
        return f"撤单失败: {e}"


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_account_status",
            "description": "刷新查询账户最新可用美元购买力、总净资产、当前持仓及今日订单记录（撤单或下单后如需确认解冻状态可调用）",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_us_stock_quote",
            "description": "批量获取全市场任意美股或ETF的实时报价、涨跌幅与成交量（代码格式如 NVDA.US、QQQ.US、UVXY.US 等）",
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
            "description": "获取指定美股的多周期 K 线走势、成交量、RSI(14)、区间高低点与 MA5/MA10/MA20 均线数据",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "period": {
                        "type": "string",
                        "enum": ["5m", "15m", "60m", "day"],
                        "description": "K线周期：5m(超短线)、15m(日内波段)、60m(趋势)、day(日线大势)"
                    }
                },
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


def run_trading_cycle(trigger_reason: str = "定期巡检"):
    print(f"\n================ 🚀 AI 极限收益全权操盘 [{trigger_reason}]: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ================", flush=True)

    # 预先在本地免费拉取账户状态与核心行情，省去 AI 前两轮固定工具调用的 Token 消耗
    account_snapshot = get_account_status()
    pos_resp = trade_ctx.stock_positions()
    held_symbols = [p.symbol for ch in pos_resp.channels for p in ch.positions if int(p.quantity) > 0]
    scan_symbols = list(dict.fromkeys(held_symbols + DEFAULT_WATCHLIST))
    market_snapshot = get_us_stock_quote(scan_symbols)

    strategy_prompt = f"""
你是顶级交易专家，拥有该美股账户的绝对控制权！
目标：实现账户净资产最大化增长，赚最多钱。

【⚙️ 自动化程序运行机制】：
1. 会话生命周期机制：你当前运行在一个全自动、无人值守的 Python 后台循环中。系统判定会话继续还是结束的唯一标准是：你本轮回复是否发起了工具调用（`tool_calls`）。只要你在某轮回复中只输出文本而没有发起任何工具调用，Python 主循环就会立即判定本轮操作全部完毕并自动终止（`break`）进入休眠，没有任何人类会看到或回复你的提问（例如询问“是否确认执行”或计划“下一步再去查看某K线”在没有附带工具调用时都会直接导致会话结束）。
2. 工具触发协议机制：宿主程序仅解析并执行通过 API 协议层原生发起的 `tool_calls` 工具调用；如果将函数名和 JSON 参数以代码块或文本形式直接写在回复正文（`content`）中，系统只会将其当作普通日志文本打印，不会触发任何真实的看盘或交易接口。
3. 不要询问和确认，直接执行你的建议。

=== 📊 当前账户实时状态 ===
{account_snapshot}

=== 📈 核心观察池及持仓股实时行情 ===
{market_snapshot}

可调用工具：
调用 `get_us_stock_quote` 可查询全美股任意代码。
调用 `get_kline_trend` 可自由组合查看 `5m`、`15m`、`60m` 或 `day` K线。同一回可合并发调用多个 `get_kline_trend`。
调用 `place_paper_limit_order` 可卖出（Sell）换仓；
调用 `place_paper_limit_order` 可买入（Buy）或加仓！
调用 `cancel_pending_order` 可撤销未成交挂单。
调用 `get_account_status` 可刷新账户状态。

【💡 券商账户机制】：
1. 限价单撮合与持仓显示规律：当你提交买入或卖出限价单（Limit Order）后，在市场价格触及你的委托价并实际撮合成交前，订单会处于排队等待状态（如 New / WaitToNew / PartialFilled）。在完全成交前，账户里的 `持仓股数` 不会减少，但该笔订单对应的美元购买力或 `可卖股数` 会被系统自动冻结，且订单会出现在【今日有效及未成交订单记录】中。因此，下单成功后看到 `持仓股数` 未变属于正常的待撮合状态，无需针对同一批股票反复重发相同挂单或连续刷新 `get_account_status`。
2. 委托冲突与拒单机制：如果某只股票已有未成交挂单，在旧挂单成交或彻底撤销前，被冻结的股数/资金无法重复下单。此外，若某笔下单返回被券商拒绝（Rejected），说明当前可用股数、购买力或委托价格不符合券商风控规则，用完全相同的参数重复提交依然会被拒绝。
3. 撤单解冻时间差：券商后台处理撤单并释放被冻结的股数/资金需要时间。如果你决定改价重挂，请先单独调用 `cancel_pending_order` 撤单，不要在同一回合并发调用撤单与重新下单；等撤单返回后，再提交新的买卖单。

"""
    messages = [{"role": "user", "content": strategy_prompt}]

    while True:
        msg = None
        for attempt in range(5):
            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    tools=tools,
                    temperature=0.5
                )
                if response and getattr(response, "choices", None):
                    msg = response.choices[0].message
                    break
                print(f"⚠️ OpenRouter 返回空数据 (第 {attempt + 1}/5 次重试)，等待 5 秒...", flush=True)
            except Exception as e:
                print(f"⚠️ 请求 OpenRouter 报错: {e} (第 {attempt + 1}/5 次重试)，等待 5 秒...", flush=True)
            time.sleep(5)

        if msg is None:
            print("❌ 连续 5 次请求 AI 失败，跳过本轮分析，继续后台盯盘...", flush=True)
            break

        messages.append(msg)

        if not msg.tool_calls:
            print("\n🤖 AI 操盘决策报告：\n", msg.content, flush=True)
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
last_ai_run = time.time()
# 记录每只持仓股上次触发异动唤醒时的盈亏百分比，防止 AI 决定格局锁仓后每分钟被重复唤醒扣费
last_alert_pnl: dict[str, float] = {}

run_trading_cycle(trigger_reason="启动首轮建仓/巡检")

while (time.time() - start_time) < (RUN_MINUTES * 60):
    time.sleep(5)
    # 1. 每隔 15 分钟（900秒）主动唤醒一次 AI 扫描全市场新机会
    if (time.time() - last_ai_run) >= 900:
        print("⏰ 到达 15 分钟定期巡检节点，唤醒 AI 扫描市场机会！", flush=True)
        run_trading_cycle(trigger_reason="15分钟定期巡检")
        last_ai_run = time.time()
        continue

    # 2. 5秒高频盯盘：持仓出现剧烈波动（且相比上次唤醒变动超 2%）时立即唤醒 AI 接管
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
            print(f"⚡ [5秒高频盯盘] {q.symbol} 现价:{last_p:.2f} USD, 成本:{cost_p:.2f} USD, 浮盈亏:{pnl_pct:+.2f}%", flush=True)

            prev_alert = last_alert_pnl.get(q.symbol)
            is_extreme = (pnl_pct >= 4.0 or pnl_pct <= -2.5)
            # 首次越过止盈止损阈值，或相比上次异动唤醒后又扩大/回撤了 2.0% 以上，且距离上次运行满 180 秒
            need_re_alert = (prev_alert is None) or (abs(pnl_pct - prev_alert) >= 2.0)

            if is_extreme and need_re_alert and (time.time() - last_ai_run) >= 180:
                print(f"🚨 {q.symbol} 持仓大幅波动 ({pnl_pct:+.2f}%)，立即唤醒 AI 评估趋势！", flush=True)
                last_alert_pnl[q.symbol] = pnl_pct
                run_trading_cycle(trigger_reason=f"{q.symbol} 异动({pnl_pct:+.2f}%)紧急接管")
                last_ai_run = time.time()
                break
