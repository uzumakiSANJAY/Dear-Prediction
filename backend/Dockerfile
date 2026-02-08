FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm install --production

COPY . .

EXPOSE ${BACKEND_PORT}

CMD ["node", "src/index.js"]
