.PHONY: build start stop restart

build: stop
	docker-compose up --build

start:
	docker-compose up app

training:
	docker-compose run training

stop:
	docker-compose down

restart: stop start