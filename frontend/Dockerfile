FROM node:20-alpine

WORKDIR /app

# Dependencies first so source edits do not bust the layer cache.
COPY package.json package-lock.json ./
RUN npm ci

COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
