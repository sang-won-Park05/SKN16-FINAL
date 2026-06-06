"use client";

import { useState } from "react";
import Link from "next/link";
import Image from "next/image";

type CodeSnippetCardProps = {
  title: string;
  description: string;
  source: string;
  openLabel: string;
  closeLabel: string;
  code: string;
};

const codeSnippets: CodeSnippetCardProps[] = [
  {
    title: "API 진입",
    description: "사용자 질문을 ChatState로 정규화하고 그래프 실행 결과를 answer / sources로 반환하는 진입 흐름입니다.",
    source: "app.py / chatbot/api/router.py / chatbot/graph.py",
    openLabel: "API 진입 코드 보기",
    closeLabel: "API 진입 코드 접기",
    code: `async def chatbot_query(payload: ChatQueryRequest):
    user_id = _resolve_user_id(None)
    state: ChatState = {
        "user_id": str(user_id),
        "messages": [{"role": "user", "content": payload.query, "meta": {}}],
    }
    if payload.session_id:
        state["session_id"] = str(payload.session_id)

    result = chatbot_graph.invoke(state)
    answer = result.get("answer") or result["messages"][-1]["content"]
    sources = result.get("sources") or []

    session_id = upsert_session_with_log(
        session_id=payload.session_id or None,
        user_id=user_id,
        query=payload.query,
        answer=answer,
        sources=sources,
    )`,
  },
  {
    title: "Orchestrator 제어",
    description: "질문 의도를 1차 라우팅한 뒤 필요한 에이전트를 골라 순차 실행하는 멀티에이전트 제어 로직입니다.",
    source: "chatbot/core/supervisor.py",
    openLabel: "Orchestrator 흐름 보기",
    closeLabel: "Orchestrator 흐름 접기",
    code: `def run_orchestrator(state: ChatState) -> ChatState:
    user_message = _get_last_user_message(state)
    if not user_message:
        return chit_agent.run(state)

    primary_route = route_supervisor(state)
    planned_routes = _plan_routes_with_llm(
        user_message=user_message,
        primary_route=primary_route,
    )

    current_state = state
    for route in planned_routes:
        current_state = _run_agent(route, current_state)

    if current_state.get("answer"):
        return current_state
    return chit_agent.run(current_state)`,
  },
  {
    title: "Agent 처리",
    description: "질병/약 에이전트는 검색, 재정렬, 신뢰도 계산을 거쳐 답변과 출처를 state에 기록합니다.",
    source: "chatbot/agents/disease_agent.py",
    openLabel: "Agent 처리 코드 보기",
    closeLabel: "Agent 처리 코드 접기",
    code: `@traceable(name="disease_agent")
def run(state: ChatState) -> ChatState:
    query = state["messages"][-1]["content"]
    docs_pool = search_disease_docs(query, pool_size=50)  # disease + interaction
    ranked = rerank(query=query, docs=[doc["text"] for doc in docs_pool], top_k=5)
    q_score = compute_qscore(ranked, query=query)

    context_text, sources = build_context_and_sources(ranked, docs_pool)
    if q_score < 0.15:
        context_text, sources = add_web_fallback(query, context_text, sources)

    answer = call_llm(
        system_prompt=DISEASE_SYSTEM_PROMPT.format(q_score=q_score),
        user_message=query,
        context=context_text,
    )

    state["answer"] = answer
    state["sources"] = sources
    return state`,
  },
];

function CodeSnippetCard({
  title,
  description,
  source,
  openLabel,
  closeLabel,
  code,
}: CodeSnippetCardProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <article className="glass rounded-3xl border-slate-200/60 p-6 md:p-7 transition-all hover:border-slate-300/70">
      <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
        <div className="min-w-0">
          <div className="text-[11px] font-mono uppercase tracking-[0.24em] text-[#4f6fc2]/70 mb-3">
            {source}
          </div>
          <h3 className="text-xl font-bold text-slate-900">{title}</h3>
          <p className="mt-2 text-sm leading-relaxed text-slate-600 max-w-2xl">{description}</p>
        </div>
        <button
          type="button"
          onClick={() => setIsOpen((prev) => !prev)}
          className="inline-flex shrink-0 items-center justify-center gap-2 rounded-2xl border border-[#4f6fc2]/20 bg-[#4f6fc2]/8 px-4 py-2 text-sm font-semibold text-[#4f6fc2] transition-all hover:bg-[#4f6fc2]/12 hover:border-[#4f6fc2]/30"
          aria-expanded={isOpen}
        >
          {isOpen ? closeLabel : openLabel}
          <span className={`text-xs transition-transform ${isOpen ? "rotate-180" : ""}`}>⌄</span>
        </button>
      </div>

      {isOpen && (
        <div className="mt-5 overflow-hidden rounded-2xl border border-slate-800/70 bg-slate-950/95 shadow-inner">
          <div className="flex items-center justify-between border-b border-slate-800/80 px-4 py-3">
            <span className="text-[11px] font-mono uppercase tracking-[0.24em] text-slate-400">
              Portfolio Snippet
            </span>
            <span className="text-xs text-slate-500">축약 코드</span>
          </div>
          <pre className="max-h-80 overflow-x-auto overflow-y-auto px-4 py-4 text-[13px] leading-6 text-slate-200">
            <code className="whitespace-pre">{code}</code>
          </pre>
        </div>
      )}
    </article>
  );
}

export default function Project1Page() {
  return (
    <main className="min-h-screen relative text-foreground selection:bg-[#4f6fc2]/20">
      {/* 2. 네비게이션 바 */}
      <nav className="fixed top-0 w-full z-50 glass border-b border-slate-200/60 px-6 py-4">
        <div className="max-w-6xl mx-auto flex justify-between items-center">
          <Link href="/#project" className="group flex items-center gap-2 text-slate-600 hover:text-[#4f6fc2] transition-colors">
            <span className="transition-transform group-hover:-translate-x-1">←</span>
            <span className="font-medium">돌아가기</span>
          </Link>
          <div className="text-xs font-mono text-slate-500 tracking-widest uppercase">Project #01</div>
        </div>
      </nav>

      <div className="max-w-6xl mx-auto px-6 pt-24 pb-16 md:px-10 md:pt-32 md:pb-24 lg:px-12">
        {/* 3. 히어로 헤더 */}
        <header className="mb-16">
          <div className="inline-block px-3 py-1 mb-6 rounded-full glass border-slate-200/60 text-xs font-bold text-[#4f6fc2] uppercase tracking-tighter">
            Healthcare Solution
          </div>
          <h1 className="text-6xl md:text-8xl font-black mb-8 tracking-tighter">Medinote</h1>
          <p className="text-xl md:text-2xl text-slate-600 max-w-3xl leading-relaxed font-light">
            흩어져 있는 의료 기록을 한곳에 모으고, <span className="text-slate-900 font-medium">복약 정보를 정리·확인</span>할 수 있도록 돕는 서비스입니다.
          </p>
        </header>

        {/* 4. 대형 이미지 섹션 */}
        <section className="mb-24">
          <div className="glass aspect-video rounded-[2rem] border-slate-200/60 overflow-hidden relative group shadow-2xl p-4 md:p-12">
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-2/3 h-2/3 bg-[#4f6fc2]/10 blur-[120px] -z-10" />
            <div className="relative w-full h-full rounded-2xl border border-slate-200/60 bg-slate-50 p-4 backdrop-blur-sm overflow-hidden">
              <Image
                src="/medinote_image/medinote_메인페이지.png"
                alt="Medinote Main"
                fill
                sizes="(min-width: 1024px) 960px, 100vw"
                className="w-full h-auto object-contain opacity-85 transition-all duration-700 group-hover:opacity-100 group-hover:scale-[1.01]"
              />
            </div>
            <div className="absolute inset-0 bg-gradient-to-t from-slate-200/40 to-transparent pointer-events-none" />
          </div>
        </section>

        {/* 5. 상세 내용 그리드 */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-16">
          <div className="lg:col-span-2 space-y-20">
            {/* 소개 (프로젝트 소개) */}
            <div>
              <h2 className="text-3xl font-bold mb-8 text-[#4f6fc2]">소개</h2>
              <p className="text-lg text-slate-700 leading-loose max-w-2xl break-keep">
                Medinote는 흩어져 있는 처방전과 진료 기록을 구조화된 데이터로 정리하고,
                복약 이력을 체계적으로 관리할 수 있도록 설계한 의료 정보 관리 서비스입니다.
                단일 RAG 구조의 한계를 개선하기 위해 Supervisor 기반 멀티 에이전트 아키텍처를 도입해
                질병·약·비의료 질문을 분리 처리하며, 안전한 의료 정보 제공을 위한 가드레일을 함께 설계했습니다.
              </p>
            </div>

            {/* 핵심 기능 */}
            <div>
              <h2 className="text-3xl font-bold mb-8 text-[#4f6fc2]">핵심 기능</h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {[
                  { title: "자동 복약 기록", desc: "처방전과 약 사진을 찍으면 OCR이 핵심 정보를 추출하여 자동으로 복약 리스트를 생성합니다." },
                  { title: "상호작용 위험 경고", desc: "병용 금기나 주의 사항을 데이터베이스 기반으로 분석해 사용자에게 즉각적인 주의 메시지를 전달합니다." },
                  { title: "AI 건강 상담 챗봇", desc: "사용자의 과거 복약 이력과 진료 기록을 바탕으로 맞춤형 건강 질문에 답변합니다." },
                  { title: "데이터 시각화", desc: "복잡한 의료 데이터를 로드맵 형태로 요약하여 사용자가 자신의 건강 흐름을 이해하도록 돕습니다." }
                ].map((item) => (
                  <div key={item.title} className="glass p-8 rounded-3xl border-slate-200/60 hover:border-slate-300/70 transition-all group">
                    <h3 className="font-bold text-xl mb-3 group-hover:text-[#4f6fc2] transition-colors">{item.title}</h3>
                    <p className="text-slate-600 leading-relaxed">{item.desc}</p>
                  </div>
                ))}
              </div>
            </div>

            {/* 아키텍처 섹션 */}
            <div>
              <h2 className="text-3xl font-bold mb-8 text-[#4f6fc2]">아키텍처</h2>
              <div className="glass rounded-3xl border-slate-200/60 p-8 overflow-hidden">
                <div className="aspect-[16/9] rounded-2xl border border-slate-200/60 bg-slate-50 flex items-center justify-center overflow-hidden">
                  <Image
                    src="/medinote_image/medinote_아키텍처.drawio.png"
                    alt="Architecture"
                    width={1600}
                    height={900}
                    sizes="(min-width: 1024px) 600px, 100vw"
                    className="w-full h-auto object-contain"
                  />
                </div>
                <p className="mt-6 text-sm text-slate-600 leading-relaxed">
                  * OCR·LLM·STT 처리는 Redis Queue 기반의 <span className="text-slate-700 font-medium">비동기 파이프라인</span>으로 구성해, 요청이 몰리는 상황에서도 메인 API 응답을 안정적으로 유지하도록 설계했습니다.
                </p>
              </div>
            </div>

            {/* 핵심 코드 스니펫 */}
            <div>
              <h2 className="text-3xl font-bold mb-4 text-[#4f6fc2]">핵심 코드 스니펫</h2>
              <p className="mb-8 max-w-2xl text-sm leading-relaxed text-slate-600">
                실제 <span className="font-medium text-slate-700">AI_service_LLM</span> 폴더의 흐름을 기준으로, API 진입부터 오케스트레이션과 에이전트 처리까지 포트폴리오용 핵심 코드만 축약해 담았습니다.
              </p>
              <div className="grid grid-cols-1 gap-5">
                {codeSnippets.map((snippet) => (
                  <CodeSnippetCard key={snippet.title} {...snippet} />
                ))}
              </div>
            </div>

            {/* 문제점 & 해결방법 */}
            <div>
              <h2 className="text-3xl font-bold mb-8 text-[#4f6fc2]">문제점 & 해결방법</h2>
              <div className="space-y-6">
                {[
                  {
                    title: "비정형 의료 데이터의 체계적인 정규화",
                    problem: "약물·질병 데이터 출처별 구조와 표현 방식 상이 → 서비스 활용 어려움",
                    details: [
                      "AIHub 의사–환자 대화 기반 질병 데이터 → 서비스 구조에 맞춘 재정리 필요",
                      "약품 데이터 전문 용어·불필요 정보 다수 → 검색·매칭 품질 저하",
                      "상품명·성분명·주의사항 표현 불일치 → 동일 의미 데이터 중복 인식 문제"
                    ],
                    solution: "질병·약품 데이터 분리 전처리 전략 및 약품 중심 정규화 파이프라인 구축",
                    solutions: [
                      "약물·성분·주의사항·금기·병용주의 기준 데이터 구조 통일",
                      "장문 주의사항 의미 단위 분리 및 핵심 정보 재구성",
                      "질병·성분·위험도 태그 추가 → 챗봇 검색 정확도 개선"
                    ],
                    outcome: "근거 기반 의료 정보 제공 가능한 정규화 데이터 구조 확보"
                  },
                  {
                    title: "단일 RAG 구조의 한계와 멀티 에이전트 아키텍처 전환",
                    problem: "하나의 RAG 파이프라인으로 모든 질문 처리 → 도메인 혼선 및 확장성 한계",
                    details: [
                      "질병·약·비의료·이력조회·외부검색 질문이 동일한 검색 전략으로 처리됨",
                      "약 질문이 질병 정보로 연결되는 등 도메인 혼선 발생",
                      "프롬프트 수정 시 전체 흐름에 영향 → 유지보수 어려움",
                      "질문 유형별 다른 처리 전략 적용 불가"
                    ],
                    solution: "LangGraph 기반 Supervisor + 도메인별 멀티 에이전트 구조 도입",
                    solutions: [
                      "Supervisor가 질문 의도 분석 후 적절한 에이전트로 라우팅",
                      "DB·History·WebSearch·질병·약·비의료 에이전트 역할 분리",
                      "LangGraph prebuilt agent(create_react_agent) 기반 구조 구현",
                      "도메인별 독립적 프롬프트·검색 전략 적용 가능 구조 설계"
                    ],
                    outcome: "질문 유형별 처리 경로 분리 및 확장 가능한 RAG 아키텍처 확보"
                  },
                  {
                    title: "안전한 의료 정보 제공을 위한 LLM 가드레일 설계",
                    problem: "의료 챗봇 답변의 법적·윤리적 위험 관리 필요",
                    details: [
                      "진단·처방 오인 가능성 존재",
                      "복용 여부 결정 등 고위험 질문 대응 체계 부재",
                      "정보 유용성 vs 답변 안전성 균형 필요"
                    ],
                    solution: "프롬프트·응답 템플릿 기반 답변 범위 제한 가드레일 설계",
                    solutions: [
                      "답변 구조 고정: 주의 고지 → 정보 안내 → 체크포인트 → 전문가 상담 권유",
                      "위험 의도 감지 시 제한 응답 출력 로직 적용",
                      "근거 없는 추측성 답변 생성 차단"
                    ],
                    outcome: "법적 리스크 최소화 및 안전 중심 의료 챗봇 UX 확보"
                  }
                ].map((item, idx) => (
                  <div key={idx} className="glass p-8 rounded-3xl border-slate-200/60 hover:border-slate-300/70 transition-all">
                    <div className="text-xs font-mono text-slate-500 tracking-widest uppercase mb-3">CASE {String(idx + 1).padStart(2, "0")}</div>
                    <h3 className="text-xl font-bold text-slate-900 mb-4">{item.title}</h3>
                    <p className="text-slate-800 leading-relaxed"><span className="font-bold text-slate-900">문제</span> <span className="text-slate-500">—</span> <span className="text-slate-700">{item.problem}</span></p>
                    <ul className="mt-4 space-y-2 text-sm text-slate-600 list-disc pl-5">
                      {item.details.map((d, i) => <li key={i}>{d}</li>)}
                    </ul>
                    <div className="h-px bg-slate-200/60 my-6" />
                    <p className="text-slate-800 leading-relaxed"><span className="font-bold text-slate-900">해결</span> <span className="text-slate-500">—</span> <span className="text-slate-700">{item.solution}</span></p>
                    <ul className="mt-4 space-y-2 text-sm text-slate-600 list-disc pl-5">
                      {item.solutions.map((s, i) => <li key={i}>{s}</li>)}
                    </ul>
                    <div className="mt-6 rounded-2xl border border-slate-200/60 bg-slate-50 p-5">
                      <div className="text-xs font-mono text-[#4f6fc2]/40 tracking-widest uppercase mb-2">Outcome</div>
                      <p className="text-sm text-slate-700 leading-relaxed">{item.outcome}</p>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* 결과/성과 */}
            <div>
              <h2 className="text-3xl font-bold mb-8 text-[#4f6fc2]">성과</h2>
              <div className="glass rounded-3xl border-slate-200/60 p-10">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-center">
                  {[
                      {
                      value: "Multi-Agent",
                      label: "멀티 에이전트 구조 설계",
                      sub: "Supervisor가 질문을 분석해 도메인별 에이전트로 라우팅하는 구조 구현"
                    },
                    {
                      value: "정확도 ⬆️",
                      label: "LLM 응답 정확도 개선",
                      sub: "프롬프트 구조화 및 RAG 적용으로 응답 품질 향상"
                    },
                    {
                      value: "토큰비용⬇️",
                      label: "인프라 및 API 토큰 최적화",
                      sub: "프롬프트 압축과 캐싱 전략으로 토큰 사용량 절감"
                    }
                  ].map((m) => (
                    <div key={m.label} className="py-2">
                      <div className={`
                        font-black text-slate-900 leading-tight
                        /* 글자 수가 길어지면 크기를 더 줄여서 한 줄 유지 */
                        ${m.value.length > 10 ? 'text-2xl md:text-3xl' : m.value.length > 7 ? 'text-3xl md:text-4xl' : 'text-4xl md:text-5xl'} 
                        /* 정확도나 비용이 포함되면 무조건 줄바꿈 방지 */
                        ${(m.value.includes('정확도') || m.value.includes('비용')) ? 'whitespace-nowrap' : ''}
                      `}>
                        {m.value}
                      </div>
                      <div className="mt-3 text-sm text-[#4f6fc2] font-semibold">{m.label}</div>
                      <div className="mt-2 text-xs text-slate-500 break-keep">{m.sub}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* 오른쪽: 사이드바 정보 */}
          <aside className="space-y-8">
            <div className="glass p-10 rounded-[2.5rem] border-slate-200/60 sticky top-32">
              <h3 className="text-sm font-bold text-slate-500 uppercase tracking-widest mb-8 text-center">Project Details</h3>
              <div className="space-y-6 mb-10 text-sm">
                <div className="flex justify-between border-b border-slate-200/60 pb-4"><span className="text-slate-500">인원</span><span className="font-medium">팀 프로젝트 (4인)</span></div>
                <div className="flex justify-between border-b border-slate-200/60 pb-4"><span className="text-slate-500">기간</span><span className="font-medium">2025. 10 - 2025. 12</span></div>
                <div className="flex justify-between border-b border-slate-200/60 pb-4"><span className="text-slate-500">역할</span><span className="font-medium">LLM,OCR 모델 설계 · 백엔드 보조</span></div>
              </div>
              <a href="https://github.com/sang-won-Park05/SKN16-FINAL.git" target="_blank" rel="noopener noreferrer" className="block w-full py-4 glass bg-slate-100 hover:bg-slate-200 text-slate-900 rounded-2xl font-bold transition-all text-center">View Source Code</a>
            </div>
          </aside>
        </div>
      </div>
      <footer className="py-20 text-center opacity-30 text-xs tracking-widest">PLAN BY SANGWON PARK</footer>
    </main>
  );
}
