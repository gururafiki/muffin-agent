FROM node:22-slim AS builder

RUN corepack enable && corepack prepare pnpm@latest --activate

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
RUN git clone --depth 1 https://github.com/langchain-ai/agent-chat-ui.git .

RUN pnpm install --frozen-lockfile

ARG NEXT_PUBLIC_API_URL=http://localhost:8123
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL

RUN pnpm build

FROM node:22-slim

RUN corepack enable && corepack prepare pnpm@latest --activate

WORKDIR /app
COPY --from=builder /app/.next .next
COPY --from=builder /app/public public
COPY --from=builder /app/package.json .
COPY --from=builder /app/pnpm-lock.yaml .
COPY --from=builder /app/node_modules node_modules

EXPOSE 3000

CMD ["pnpm", "start"]
