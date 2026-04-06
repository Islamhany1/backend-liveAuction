package com.liveauction.backend_liveAuction.repositories;

import com.liveauction.backend_liveAuction.entities.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.Optional;

@Repository
public interface UserRepository extends JpaRepository<User, Integer> {

    // Spring magically writes the SQL for this just by reading the method name!
    Optional<User> findByUsername(String username);

}